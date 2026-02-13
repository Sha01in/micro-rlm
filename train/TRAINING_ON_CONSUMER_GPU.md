# LoRA Fine-Tuning with JAX/EasyDel on a Single Consumer GPU

**A practitioner's log of getting a JAX-based LoRA SFT pipeline running end-to-end on an RTX 4070 Ti (12 GB).**

This documents the full journey of building and debugging a LoRA fine-tuning
pipeline for Qwen2.5 models using JAX, Flax NNX, EasyDel, and Orbax on a
single consumer GPU. We encountered and resolved six distinct failure modes,
ranging from XLA compiler crashes to a hidden 1 GB memory allocation buried
inside EasyDel's attention layer. Every fix is explained with the root cause,
the diagnosis that led us there, and the exact code change.

**Hardware:** NVIDIA RTX 4070 Ti (12 GB VRAM), WSL2 on Windows
**Software:** Python 3.10, JAX 0.6.2, EasyDel 0.1.4.1, Flax 0.10.4
**Task:** LoRA SFT on Qwen2.5-0.5B/1.5B using RLM trajectory data
**Result:** Full training pipeline working end-to-end in ~5 GB VRAM (0.5B model)

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [The dependency maze](#2-the-dependency-maze)
3. [Bug 1: XLA scheduler crash (jaxlib 0.4.34)](#3-bug-1-xla-scheduler-crash)
4. [Bug 2: "No mesh found" — EasyDel's hidden sharding requirement](#4-bug-2-no-mesh-found)
5. [Bug 3: LoRA injection finds zero layers](#5-bug-3-lora-injection-finds-zero-layers)
6. [Bug 4: 3.5 GB MLIR constant embedding OOM](#6-bug-4-mlir-constant-embedding-oom)
7. [Bug 5: The hidden 1 GB causal mask](#7-bug-5-the-hidden-1-gb-causal-mask)
8. [Bug 6: Death by a thousand small GPU allocations](#8-bug-6-death-by-a-thousand-small-allocations)
9. [The working configuration](#9-the-working-configuration)
10. [GPU memory budget reference](#10-gpu-memory-budget-reference)
11. [Lessons learned](#11-lessons-learned)

---

## 1. Architecture Overview

The training pipeline fine-tunes a pre-trained Qwen2.5 model using LoRA
(Low-Rank Adaptation) to learn the behavioral patterns of a Recursive
Language Model (RLM). The pipeline has three training scripts sharing common
infrastructure:

| Script | Purpose | Model |
|---|---|---|
| `sft_root.py` | Teach the model to *be* the RLM (write REPL code, make sub-calls, emit answers) | Qwen2.5-0.5B (default) |
| `sft_sub.py` | Specialist for sub-call tasks (analyze chunks, count items) | Qwen2.5-0.5B |
| `reward_model.py` | Score trajectory quality for data filtering | Qwen2.5-0.5B + reward head |

**Stack:** EasyDel loads HuggingFace models as Flax NNX modules. We inject
LoRA layers, split the model into frozen base + trainable LoRA params using
`nnx.split`, and train with `jax.value_and_grad` in a functional style.
Checkpoints are saved with Orbax.

```
EasyDel (model loading) → Flax NNX (module system) → JAX (JIT/autodiff) → XLA (GPU compilation)
```

This stack is powerful but has sharp edges. Each layer has its own assumptions
about memory, sharding, and compilation that interact in non-obvious ways.

---

## 2. The Dependency Maze

Getting a consistent set of JAX ecosystem packages that all work together on
Python 3.10 with CUDA 12 is its own challenge. Here is the final working set
and why each version is pinned:

```
jax[cuda12]==0.6.2          # 0.5.0+ fixes XLA crash; 0.7+ drops Python 3.10
jaxlib==0.6.2               # must match jax version
flax==0.10.4                # NNX API; requires jax>=0.4.27
optax==0.2.5                # 0.2.6+ requires jax>=0.5.3
orbax-checkpoint==0.11.32   # 0.7.0 broken with JAX 0.6 (see Bug 6)
chex==0.1.90                # 0.1.91+ requires jax>=0.7.0
easydel==0.1.4.1            # pinned — API changes every release
transformers==4.57.6        # must be 4.x — EasyDel incompatible with 5.x
```

**Key constraint:** Python 3.10 caps JAX at 0.6.2 (JAX 0.7+ requires
Python 3.11). This creates a narrow compatibility window where every package
must be individually version-checked.

---

## 3. Bug 1: XLA Scheduler Crash

**Symptom:** Model compilation crashes with a CHECK failure deep inside XLA.

```
gpu_hlo_schedule.cc:475] Check failed:
  collective_broadcast_overlap_limit <= parallel_collective_overlap_limit
```

**Root cause:** A known bug in jaxlib 0.4.34/0.4.35
([jax-ml/jax#25404](https://github.com/jax-ml/jax/issues/25404)). EasyDel
injects XLA flags at import time (`--xla_gpu_enable_latency_hiding_scheduler=true`,
`--xla_gpu_enable_pipelined_all_gather=true`, etc.) that trigger a code path
in the GPU HLO scheduler with an off-by-one bounds check. Simple operations
like matrix multiplication work fine — the crash only manifests when compiling
large model graphs that exercise the collective scheduling logic.

**Diagnosis path:** We tried every conceivable XLA flag override
(`--xla_gpu_enable_latency_hiding_scheduler=false`, etc.), but EasyDel
re-injects its flags after our overrides. The flags are set at the Python
level in EasyDel's `__init__.py` and there's no API to disable them.

**Fix:** Upgrade JAX from 0.4.35 to 0.6.2. The bug was fixed upstream in
JAX 0.5.0+.

```bash
uv pip install "jax[cuda12]==0.6.2"
```

**Lesson:** When an XLA crash cites an internal scheduler assertion, check the
JAX issue tracker before debugging flags. XLA flags are a red herring when the
bug is in the compiler itself.

---

## 4. Bug 2: "No Mesh Found"

**Symptom:** After upgrading JAX, model loading works but the forward pass
immediately fails:

```
ValueError: No mesh found under this context manager.
```

**Root cause:** EasyDel 0.1.x uses the `eformer` sharding layer internally,
which requires a JAX `Mesh` context even on a single GPU. Previous JAX versions
were apparently more lenient about this requirement.

**Fix:** Create a trivial single-device mesh and wrap all model operations:

```python
from jax.sharding import Mesh
import numpy as np

def make_mesh():
    return Mesh(np.array(jax.devices()), ('dp',))

mesh = make_mesh()
with mesh:
    output = model(input_ids=ids, attention_mask=mask)
```

This must wrap every `model(...)` call, including warmup, training steps, and
inference.

---

## 5. Bug 3: LoRA Injection Finds Zero Layers

**Symptom:** `inject_lora()` reports 0 layers injected, even though the model
clearly has attention and MLP layers.

**Root cause:** Our LoRA injection code walked the module tree looking for
`nnx.Linear` layers. But EasyDel doesn't use `nnx.Linear` — it uses its own
`ParallelLinear` class that has the same interface (a `.kernel` attribute) but
doesn't inherit from `nnx.Linear`.

```python
# This finds 0 layers:
isinstance(layer, nnx.Linear)  # False for EasyDel's ParallelLinear

# This finds 196 layers:
isinstance(layer, nnx.Module) and hasattr(layer, 'kernel')
```

**Fix:** Replace the type check with a duck-typing check:

```python
def _is_linear(layer):
    if isinstance(layer, nnx.Linear):
        return True
    return isinstance(layer, nnx.Module) and hasattr(layer, 'kernel')
```

After the fix, we correctly find and inject LoRA into all 7 target modules
(q/k/v/o_proj + gate/up/down_proj) across all transformer layers:
- Qwen2.5-0.5B: 168 layers, 8.8M LoRA params
- Qwen2.5-1.5B: 196 layers, 18.5M LoRA params

---

## 6. Bug 4: MLIR Constant Embedding OOM

**Symptom:** Training step compilation crashes trying to allocate 3.55 GB
during MLIR lowering, before any actual computation:

```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 3554312704 bytes
```

**Root cause:** When you capture a large pytree in a `@jax.jit` closure, XLA
embeds it as constant data in the compiled MLIR program. The frozen base model
weights (~3.5 GB for Qwen2.5-1.5B) were captured this way:

```python
# BAD: base_state captured in closure → embedded as 3.5 GB MLIR constants
@jax.jit
def train_step(lora_state, opt_state, batch):
    def loss_fn(ls):
        model = nnx.merge(graphdef, ls, base_state)  # base_state from outer scope
        ...
```

**Fix:** Pass `base_state` as an explicit argument. JAX then treats it as
device data rather than program constants:

```python
# GOOD: base_state as argument → stays as device data
@jax.jit
def train_step(lora_state, opt_state, base_state, batch):
    def loss_fn(ls):
        model = nnx.merge(graphdef, ls, base_state)
        ...
```

Note: `graphdef` can remain in the closure because it's just a Python data
structure describing the module hierarchy, with no large arrays.

**Related issue with `jax.checkpoint`:** You can't pass the reconstructed NNX
model as an argument to `jax.checkpoint` because NNX modules aren't valid JAX
types during tracing:

```python
# BAD: DynamicJaxprTracer is not callable
jax.checkpoint(lambda m, ids, mask: m(...))(model, ids, mask)

# GOOD: model stays in closure, only arrays cross the checkpoint boundary
model = nnx.merge(graphdef, ls, base_state)
jax.checkpoint(lambda ids, mask: model(input_ids=ids, attention_mask=mask))(ids, mask)
```

---

## 7. Bug 5: The Hidden 1 GB Causal Mask

**Symptom:** Even after all previous fixes, a forward-only pass on
Qwen2.5-0.5B (a 1.3 GB model) consumes 6.1 GB of GPU memory for a
sequence of just 16 tokens. This leaves no room for backward pass.

**Diagnosis:** We used `nvidia-smi` at each stage to create a memory profile:

```
After model load:    4807 MiB used  (model weights)
After forward pass: 10941 MiB used  (!!!)
Delta:               6134 MiB for seq_len=16, batch_size=1
```

6 GB for a forward pass on 0.5B with 16 tokens is absurdly high. We inspected
the EasyDel source code and found the culprit in the base model class:

```python
# easydel/infra/base_config.py
def get_basic_causal_mask(self):
    target_length = self.granted_mask_max_position_embedding  # 32768 for Qwen2.5
    return self._create_causal_mask(target_length)
    # Creates a (1, 1, 32768, 32768) boolean tensor = 1.07 GB
```

**Root cause:** EasyDel precomputes the causal attention mask at model
initialization, allocating it for the model's *maximum* position embedding
length (32,768 for Qwen2.5). This produces a 32K x 32K boolean tensor = **1.07 GB**
on the GPU, regardless of the actual sequence length being processed. The
mask is stored as a cached property on the model and materialized on first access.

The remaining ~5 GB came from XLA's eager execution mode not reusing buffers
(we weren't JIT-compiling the forward pass during diagnosis).

**Fix:** One line:

```python
model.config.precompute_masks = False
```

This tells EasyDel to skip precomputation and instead compute causal masks
on-the-fly for each forward pass, sized to the actual sequence length. For
seq_len=1024, the mask is 1024x1024 = 1 MB instead of 1 GB.

**Impact:** With this fix plus JIT compilation, forward pass memory dropped
from 6134 MiB to **1189 MiB** — a 5x reduction. Combined with the previous
fixes, we went from "won't even compile" to having 5.9 GB free after a
forward pass, plenty for backward.

---

## 8. Bug 6: Death by a Thousand Small Allocations

Several smaller issues caused OOM at unexpected points after the major fixes:

### 8a. Orbax `enable_memories` AttributeError

After training completes successfully, saving the checkpoint fails:

```
AttributeError: module 'jax._src.config' has no attribute 'enable_memories'
```

**Cause:** Orbax-checkpoint 0.7.0 references a JAX internal API
(`jax._src.config.enable_memories`) that was removed in JAX 0.6.x because the
feature it controlled became permanently enabled.

**Fix:** Upgrade orbax-checkpoint from 0.7.0 to 0.11.32.

### 8b. JAX RNG Permutation OOM

After model loading, a simple data shuffling operation OOMs:

```python
indices = jrandom.permutation(shuffle_rng, len(samples)).tolist()
# RESOURCE_EXHAUSTED: out of memory
```

**Cause:** `jax.random.permutation` allocates a JAX array on the GPU, then
`.tolist()` copies it back to CPU. When GPU memory is nearly full (as it is
after loading a 3.6 GB model with 3.3 GB baseline), even this tiny allocation
fails because JAX's memory allocator can't find a free fragment.

**Fix:** Use numpy instead of JAX for shuffling — it runs entirely on CPU:

```python
np_rng = np.random.default_rng(seed)
indices = np_rng.permutation(len(samples)).tolist()
```

### 8c. Model reference leak after split

After `nnx.split(model, LoRAParam, ...)`, the original `model` variable
still references the NNX module graph, potentially keeping non-shared buffers alive.

**Fix:** `del model` immediately after the split.

---

## 9. The Working Configuration

After resolving all six issues, training runs end-to-end:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python train/sft_root.py \
    --data ./train/traj_synthetic \
    --output ./models/root_v1 \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_rank 16 --lora_alpha 32.0 \
    --epochs 2 --batch_size 1 --grad_accum 2 \
    --seq_len 1024 --lr 2e-4
```

Output:

```
Loading trajectories from ./train/traj_synthetic
  Loaded 12 RLM trajectories
  After correctness filter: 10

Tokenizing with Qwen/Qwen2.5-0.5B-Instruct tokenizer
  10 tokenized samples (max_len=1024)
  Avg sequence length: 826 tokens
  Total supervised tokens: 1,574

Loading model: Qwen/Qwen2.5-0.5B-Instruct
  Base params: 630,167,427 (1.3 GB in bf16)
  LoRA params: 8,798,208 (rank=16, alpha=32.0)
  LoRA layers injected: 168
  Trainable: 1.40%

Training: 2 epochs, 10 optimizer steps
  XLA compilation took 61.3s
  Epoch 1/2 — avg_loss=0.6265
  Epoch 2/2 — avg_loss=0.0079

Training complete: 546.7s total, best avg_loss=0.0079
  Saved LoRA weights to ./models/root_v1/lora_params
```

Peak GPU memory: ~5 GB with 0.5B model, leaving ~3.7 GB headroom on a 12 GB card.

---

## 10. GPU Memory Budget Reference

Measured on RTX 4070 Ti (12,282 MiB total), WSL2 with display server:

| Component | Memory |
|---|---|
| WSL2 / display driver baseline | ~3,300 MiB |
| **Available for ML** | **~8,900 MiB** |

| Model | Weights (bf16) | Training (seq=1024) | Fits 12 GB? |
|---|---|---|---|
| Qwen2.5-0.5B | 1.3 GB | ~5 GB total | Yes (3.9 GB free) |
| Qwen2.5-1.5B | 3.6 GB | ~9 GB at seq=512 | Barely (seq<=512 only) |
| Qwen2.5-3B | 6.6 GB | N/A | No (OOM on load) |

For the 1.5B model, training is possible but only with `seq_len<=512`. Since
the RLM system prompt consumes ~600 tokens, this leaves almost no room for the
actual trajectory content. The 0.5B model is the practical choice for 12 GB
cards.

**Scaling guidance:**
- 12 GB (RTX 4070 Ti, 3060 12GB): Qwen2.5-0.5B, seq_len=1024
- 16 GB (RTX 4080): Qwen2.5-1.5B, seq_len=1024 (estimated)
- 24 GB (RTX 3090, 4090): Qwen2.5-3B, seq_len=2048 (estimated)

---

## 11. Lessons Learned

### On the JAX ecosystem

1. **JAX version compatibility is fragile.** Every package in the JAX ecosystem
   (Flax, Optax, Orbax, Chex) has tight JAX version bounds, and they don't
   always agree. Budget time for version resolution. Pin exact versions in CI.

2. **XLA flag injection is invisible and dangerous.** Libraries like EasyDel
   can inject XLA compiler flags at import time. These flags interact with
   specific jaxlib versions in unpredictable ways. When you see XLA crashes,
   check if your libraries are setting flags behind your back.

3. **Closure capture in `@jax.jit` has real memory cost.** Any large pytree
   captured in a JIT closure is embedded as MLIR constants. For multi-GB model
   weights, this doubles memory usage during compilation. Always pass large
   state as explicit function arguments.

### On GPU memory

4. **Profile memory at every stage.** The single most useful debugging
   technique was calling `nvidia-smi` before/after each operation to build a
   memory waterfall. This instantly revealed the 6 GB forward pass anomaly
   that would have been nearly impossible to find by reading code alone.

5. **Precomputed tensors are silent killers.** EasyDel's 32K x 32K causal mask
   silently consumed 1 GB on every model load. The mask served no purpose
   during training (where sequences are much shorter than 32K). Always check
   what gets allocated at model init time, not just during forward/backward.

6. **Every GPU allocation matters at the margin.** On a 12 GB card with 3.3 GB
   driver overhead and a 3.6 GB model, you have ~5 GB for training. In this
   regime, a 1 GB causal mask, a JAX RNG permutation, or a Python reference
   keeping a buffer alive can each be the difference between OOM and success.

### On debugging methodology

7. **Isolate before optimizing.** We tested each component in isolation
   (forward-only, JIT-only, backward-only) before combining. This made it
   possible to attribute each memory spike to its specific cause.

8. **Start with the smallest model.** We debugged with 0.5B first, then
   tested 1.5B, then 3B. Each model size added constraints, and working
   up from small to large was far more productive than starting at the target
   size and hitting OOM without context.

---

## Appendix: Quick Reference for EasyDel + JAX Training

```python
# 1. Load model
model = AutoEasyDeLModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
    auto_shard_model=False,
)

# 2. Disable precomputed causal mask (saves ~1 GB)
model.config.precompute_masks = False

# 3. Inject LoRA (duck-type check for ParallelLinear)
inject_lora(model, rank=16, alpha=32.0, rng=jax.random.PRNGKey(0))

# 4. Split model, delete original
graphdef, lora_state, base_state = nnx.split(model, LoRAParam, ...)
del model

# 5. Create mesh (required by EasyDel)
mesh = Mesh(np.array(jax.devices()), ('dp',))

# 6. Training step — base_state as argument, not closure
@jax.jit
def train_step(lora_state, opt_state, base_state, batch):
    def loss_fn(ls):
        model = nnx.merge(graphdef, ls, base_state)
        output = jax.checkpoint(
            lambda ids, mask: model(input_ids=ids, attention_mask=mask),
        )(batch["input_ids"], batch["attention_mask"])
        ...

# 7. Run with mesh context
with mesh:
    lora_state, opt_state, loss = train_step(lora_state, opt_state, base_state, batch)

# 8. Use numpy for shuffling (not jax.random — avoids GPU OOM)
np_rng = np.random.default_rng(seed)
indices = np_rng.permutation(len(samples))
```

**Environment variables:**
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false    # allocate on demand
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95    # use up to 95% of GPU memory
```

---

*Built as part of [micro-rlm](https://github.com/Sha01in/micro-rlm), a
minimal implementation of Recursive Language Models (Zhang, Kraska, Khattab —
MIT CSAIL, 2026).*
