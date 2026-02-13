# From Paper to Pipeline: Building a Complete RLM Training System on a Consumer GPU

**A comprehensive technical guide to micro-rlm — from the research paper through
inference implementation through JAX-based training on a 12 GB GPU.**

*For ML engineers interested in Recursive Language Models and JAX/Flax NNX.*

---

## Table of Contents

1. [The problem and the insight](#1-the-problem-and-the-insight)
2. [Algorithm 1: the RLM loop](#2-algorithm-1-the-rlm-loop)
3. [The inference engine: micro_rlm.py](#3-the-inference-engine)
4. [From inference to training data](#4-from-inference-to-training-data)
5. [The training pipeline](#5-the-training-pipeline)
6. [Three models, one system](#6-three-models-one-system)
7. [Serving: closing the loop](#7-serving-closing-the-loop)
8. [Making it fit: consumer GPU engineering](#8-making-it-fit-consumer-gpu-engineering)
9. [What makes this different](#9-what-makes-this-different)
10. [Codebase map](#10-codebase-map)
11. [Where it goes from here](#11-where-it-goes-from-here)

---

## 1. The Problem and the Insight

LLMs degrade as prompts get longer. Ask a model to count entries in a
200-line CSV and it does fine. Ask it to count 5,000 lines and accuracy
collapses — not because the task is harder, but because the model's
attention becomes diluted across too much context. This is "context rot,"
and it gets worse as input scales. Current workarounds — summarization,
RAG, sub-agent delegation — each lose information at the boundary between
what the model sees and what it doesn't.

The [Recursive Language Models paper](https://arxiv.org/abs/2512.24601)
(Zhang, Kraska, Khattab — MIT CSAIL, 2026) proposes a different approach:
**don't feed the prompt to the model at all**. Instead, place it in a code
execution environment and let the LLM write programs to interact with it.
The model sees only constant-size metadata — the length of the context,
a short preview, and truncated summaries of its own code's output. It
stores results in REPL variables, processes slices with sub-LLM calls
inside programmatic loops, and terminates when it's ready.

The paper demonstrates this on inputs two orders of magnitude beyond the
context window. On BrowseComp-Plus (1,000 documents, 6–11M tokens),
standard GPT-5 scores 0% — the input literally doesn't fit. The RLM
variant scores 91.3%.

This document covers the complete system we built around that idea:
a zero-dependency inference engine (`micro_rlm.py`), a JAX/Flax NNX
training pipeline that fine-tunes local models on RLM trajectories, and
the practical engineering required to make it all run on a single RTX 4070
Ti with 12 GB of VRAM.

---

## 2. Algorithm 1: The RLM Loop

Here is the algorithm from the paper, annotated with the implementation's
corresponding code:

```
Input: prompt P
Output: response Y

state ← InitREPL(prompt=P)          # REPL.__init__: context stored as variable
state ← AddFunction(state, sub_RLM) # llm_query() injected into REPL namespace
hist  ← [Metadata(state)]           # system prompt + context preview (NOT P itself)

loop:
    code   ← LLM(hist)              # llm_call(history) → root model response
    state, stdout ← REPL(code)      # repl.execute(code) → stdout, had_error
    hist   ← hist ∥ Metadata(stdout) # truncate(stdout, 800) → only summary goes back
    if state[Final] is set:
        return state[Final]          # extract_final(response) → answer
```

Three design choices distinguish this from a standard coding agent:

| | Standard Agent (Algorithm 2) | RLM (Algorithm 1) |
|---|---|---|
| **Prompt** | Dumped into LLM context window | External variable in REPL |
| **Output** | Generated autoregressively | Built up in REPL variables |
| **Recursion** | Verbalized sub-task delegation | Programmatic loops of sub-LLM calls |

The most critical mechanism is **metadata truncation**. After the model's
code executes in the REPL, only a truncated summary of stdout is returned
to the model — by default, 800 characters. This isn't a performance
optimization. It's the fundamental architectural choice that makes the
system work.

Without truncation, the model's context window gradually fills with raw
output from each iteration, and the approach degenerates into Algorithm 2
(a standard agent that happens to use a REPL). With truncation, the model
is forced to store intermediate results in REPL variables, query over
slices rather than dumping everything, and build its answer
programmatically. This is what enables O(N) or O(N^2) processing of inputs
that would overflow any context window.

```python
# micro_rlm.py, lines 308-318 — the critical truncation point
meta = f"[Execution {'ERROR' if had_error else 'OK'}]\n"
meta += truncate(stdout, args.metadata_chars)
uvars = repl.user_vars()
if uvars:
    meta += f"\n[REPL variables: {uvars}]"
history.append({"role": "user", "content": meta})
```

The REPL variable listing is a subtle but important detail — it reminds
the model what state it has built up across iterations, compensating for
the truncated stdout.

---

## 3. The Inference Engine

`micro_rlm.py` is 535 lines with zero external dependencies (Python
stdlib + `urllib` for API calls). It's organized into nine numbered
sections mapping directly to the paper. Here we focus on the three most
important architectural decisions.

### The REPL

The REPL class is minimal — a persistent `exec()` namespace with two
injected objects:

```python
class REPL:
    def __init__(self, context, llm_query_fn=None):
        self.namespace = {"context": context}
        if llm_query_fn:
            self.namespace["llm_query"] = llm_query_fn
        exec("import re, math, json\n"
             "from collections import Counter, defaultdict", self.namespace)
```

`context` is the full input document (potentially millions of characters).
`llm_query` is a closure that calls the sub-model — each invocation is a
separate API call tracked in the trajectory log. By injecting it into the
namespace rather than exposing it via an API, the model can use it
naturally inside Python code: `results = [llm_query(chunk) for chunk in chunks]`.

### The System Prompt

The system prompt (Section 5) is parametric — it tells the model the
exact size of the context and the truncation limit:

```
You are an RLM (Recursive Language Model). You must answer a query about
a large context stored in a REPL environment — NOT in your context window.

Your REPL has:
- `context` — string variable with the input data ({n_chars} chars total)
- `llm_query(prompt)` — call a sub-LLM for semantic reasoning
- `print()` — view results (WARNING: output is truncated to ~{meta_chars} chars)
```

The `{n_chars}` and `{meta_chars}` placeholders are filled at runtime.
Grounding the model in the actual problem size prevents it from attempting
naive strategies (like `print(context)`) that would be truncated to uselessness.

### Trajectory Logging

The `--log` flag captures the full RLM interaction to structured JSON.
Each trajectory records the config, query, ground truth, and per-iteration
deltas: the root model's response, extracted code, execution metadata
(stdout, errors, REPL variable state), sub-call prompts and responses,
and FINAL detection results.

This uses delta-only storage — each iteration records only its own output,
not the full accumulated history. File size stays linear in the number of
iterations rather than quadratic.

The trajectory format is the bridge between inference and training. We
discuss it in the next section.

---

## 4. From Inference to Training Data

The RLM paper (Section 4) demonstrates that fine-tuning on as few as
1,000 filtered trajectories improves RLM performance by 28%. The insight
is elegant: the model's own successful interactions become training data
for better interactions. This is a form of self-play, analogous to AlphaGo
training on its own games.

### Trajectory Structure

Each trajectory JSON captures a complete RLM session:

```json
{
  "mode": "rlm",
  "task": "census",
  "config": { "seed": 42, "n_entries": 50, "metadata_chars": 800 },
  "ground_truth": "Austin: 11\nBoston: 5\n...",
  "iterations": [
    {
      "iteration": 1,
      "root_response": "I'll count people by city.\n\n```repl\ncities = ...",
      "code": "cities = {}\nfor line in context.split(\"\\n\"): ...",
      "execution": {
        "stdout": "Found 8 cities\n  Austin: 11\n...",
        "had_error": false,
        "metadata_sent": "[REPL stdout]\nFound 8 cities\n..."
      },
      "sub_calls": [
        { "prompt": "Analyze this chunk...", "response": "Seattle: 6, Austin: 11" }
      ]
    },
    {
      "iteration": 2,
      "code": "FINAL = \"\"\"Austin: 11\nBoston: 5\n...\"\"\"",
      "final": "Austin: 11\nBoston: 5\n..."
    }
  ],
  "answer": "Austin: 11\nBoston: 5\n..."
}
```

The task generators use deterministic seeded RNG, so ground truth is
always verifiable and the full context can be regenerated from the seed
without storing it in the trajectory file.

### Three Training Targets

The paper describes three enhancements that map to three training scripts:

| Enhancement | What it teaches | Training data source |
|---|---|---|
| **Root SFT** | Be the RLM — write REPL code, use `llm_query()`, emit FINAL | Multi-turn trajectory conversations |
| **Sub-call SFT** | Handle sub-calls — analyze chunks, count items, extract info | Single-turn (prompt, response) pairs from sub-calls |
| **Reward model** | Score trajectory quality — which steps are productive? | Per-iteration labels: outcome x step penalty |

Each enhancement addresses a different cost or quality bottleneck. Root
SFT eliminates root-model API costs. Sub-call SFT eliminates sub-call
costs (often 50%+ of total). The reward model enables data filtering
(train only on high-quality trajectories) and, at inference time,
early-stopping of unproductive trajectories.

---

## 5. The Training Pipeline

The training code (`train/`) is built on JAX, Flax NNX, EasyDel, Optax,
and Orbax. For readers coming from PyTorch: JAX is a numerical computing
library with automatic differentiation and JIT compilation to XLA. Flax
NNX is its module system (analogous to `torch.nn`). EasyDel loads
HuggingFace model weights into Flax NNX modules.

### 5a. Data Processing

`data_utils.py` (347 lines) transforms trajectory JSON files into
training-ready token sequences. The key challenge is reconstructing the
exact ChatML conversation from trajectory data.

**Root model data** (`traj_to_root_messages`): Rebuilds the full multi-turn
conversation — system prompt, user query, then alternating assistant code
and user metadata for each iteration. The conversation ends with an
assistant turn (the FINAL response). Context is regenerated from the seed
rather than stored.

**Sub-call data** (`extract_sub_calls`): Flattens all sub-calls across all
trajectories into standalone (user, assistant) pairs. Each sub-call is
stateless — the sub-model doesn't need the RLM conversation context.

**Reward labels** (`label_trajectories`): Creates per-iteration samples.
Each gets a label: 1.0 if the trajectory's final answer was correct, 0.0
otherwise, multiplied by a step penalty (0.3 for error steps, 0.1 for
empty output, 1.0 for normal steps). This teaches the reward model to
prefer efficient, successful trajectories.

**Tokenization with loss masking** (`tokenize_chat`): Converts ChatML
messages to token sequences using the model's chat template, then masks
non-assistant tokens with `-100` so the loss is computed only on the
model's own output:

```python
# State machine walks through tokens, enabling labels only inside
# <|im_start|>assistant\n ... <|im_end|> regions
in_assistant = False
i = 0
while i < len(input_ids):
    if not in_assistant and _subseq_match(input_ids, i, assistant_start):
        i += len(assistant_start)
        in_assistant = True
        continue
    if in_assistant and _subseq_match(input_ids, i, assistant_end):
        in_assistant = False
        i += len(assistant_end)
        continue
    if in_assistant:
        labels[i] = input_ids[i]
    i += 1
```

This is Qwen/ChatML-specific — the markers (`<|im_start|>`, `<|im_end|>`)
are tokenized and matched as subsequences in the token stream.

### 5b. LoRA via Flax NNX

We use LoRA (Low-Rank Adaptation) to fine-tune only ~1.4% of parameters.
The implementation has three components.

**LoRAParam** — a marker class that extends `nnx.Variable`:

```python
class LoRAParam(nnx.Variable):
    pass
```

In Flax NNX, every parameter is wrapped in a `Variable` subclass. By
creating a custom subclass, we can later call `nnx.split(model, LoRAParam, ...)`
to separate LoRA params from frozen base params into different pytrees.
This is how JAX handles the "freeze some parameters, train others" pattern
that PyTorch does with `requires_grad`.

**LoRALinear** — wraps an existing linear layer with a low-rank residual:

```python
class LoRALinear(nnx.Module):
    def __call__(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.value) @ self.lora_B.value * self.scale
        return base_out + lora_out
```

`lora_B` is initialized to zeros, so the LoRA contribution starts at zero
and the model begins training from its pre-trained behavior. `self.scale`
is `alpha / rank` (default 32 / 16 = 2.0), making the learning rate
roughly independent of rank.

**inject_lora** — walks the module tree and replaces target layers:

```python
for path, module in model.iter_modules():
    for attr_name in list(vars(module)):
        if attr_name not in TARGET_MODULES:  # q/k/v/o_proj, gate/up/down_proj
            continue
        layer = getattr(module, attr_name, None)
        if _is_linear(layer):
            setattr(module, attr_name, LoRALinear(layer, rank, alpha, rngs=rngs))
```

One subtlety: EasyDel doesn't use `nnx.Linear` — it uses its own
`ParallelLinear` class that has the same interface (a `.kernel` attribute)
but a different class hierarchy. The `_is_linear` check uses duck typing
(`hasattr(layer, 'kernel')`) instead of `isinstance(layer, nnx.Linear)`.
We found this the hard way — the type check silently injected zero layers.

### 5c. The JIT-Compiled Training Step

After injecting LoRA, we split the model into three parts and build a
functional training step:

```python
# Separate the model into a graph definition, trainable LoRA params,
# and frozen base params
graphdef, lora_state, base_state = nnx.split(model, LoRAParam, ...)
del model  # free original reference

@jax.jit
def train_step(lora_state, opt_state, base_state, batch):
    def loss_fn(ls):
        model = nnx.merge(graphdef, ls, base_state)
        output = jax.checkpoint(
            lambda ids, mask: model(input_ids=ids, attention_mask=mask),
        )(batch["input_ids"], batch["attention_mask"])
        logits = output.logits

        # Standard next-token loss with label masking
        shift_logits = logits[:, :-1]
        shift_labels = batch["labels"][:, 1:]
        log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
        token_losses = -jnp.take_along_axis(
            log_probs, shift_labels[:, :, None], axis=-1
        ).squeeze(-1)
        mask = (shift_labels != -100).astype(jnp.float32)
        return (token_losses * mask).sum() / mask.sum().clip(min=1)

    loss, grads = jax.value_and_grad(loss_fn)(lora_state)
    updates, new_opt_state = optimizer.update(grads, opt_state, lora_state)
    new_lora = optax.apply_updates(lora_state, updates)
    return new_lora, new_opt_state, loss
```

Two details that matter:

**`base_state` is a function argument, not a closure capture.** When a
large pytree is captured in a `@jax.jit` closure, JAX embeds it as
constant data in the compiled MLIR program. For a 1.5B model (3.5 GB of
weights), this doubles memory during compilation — an OOM that manifests
during MLIR lowering, before any actual computation. Passing `base_state`
as an argument tells JAX to treat it as device data, avoiding the copy.

**`jax.checkpoint` wraps the forward pass with the model in a closure.**
Gradient checkpointing recomputes forward activations during the backward
pass instead of storing them. The tricky part: NNX modules aren't valid
JAX types during tracing, so you can't pass the model as an argument to
`jax.checkpoint`. Instead, the reconstructed model lives in the closure
(which is fine — it's just a module reference, not a large array), and
only JAX arrays (`input_ids`, `attention_mask`) cross the checkpoint
boundary.

---

## 6. Three Models, One System

### Root Model SFT (`sft_root.py`, 402 lines)

The root model learns the full RLM interaction pattern: peek at the
context structure, design a chunking strategy, process chunks with
`llm_query()`, store intermediate results in REPL variables, and emit
`FINAL` or `FINAL_VAR` when the answer is complete.

Training data is multi-turn: each trajectory becomes a conversation
of 4–20+ messages alternating between assistant code and user execution
metadata. The loss is computed only on assistant tokens — the model learns
to generate code and FINAL markers, not to predict what the REPL outputs.

Verified results on Qwen2.5-0.5B-Instruct (630M base params):
- **LoRA:** 168 layers, 8.8M trainable params (1.4% of total)
- **Training:** 2 epochs on 10 trajectories, 10 optimizer steps
- **Loss:** 0.63 → 0.008 (converges quickly on small data)
- **Time:** 547 seconds total (61s XLA compilation + 486s training)
- **VRAM:** ~5 GB peak on 12 GB RTX 4070 Ti

### Sub-Call Specialist (`sft_sub.py`, 279 lines)

Sub-calls are simple, stateless tasks: "analyze this chunk and count
people per city," "extract mentions of Project X from these documents."
A small fine-tuned model handles these well.

The training data is single-turn — each sub-call from every trajectory
becomes a standalone (user prompt, assistant response) pair. No
multi-turn context needed. The sub-model doesn't need to understand the
RLM paradigm; it just needs to follow instructions on data chunks.

Different hyperparameters reflect the simpler task: lower learning rate
(5e-5 vs 2e-4), more epochs (5 vs 3), longer sequence length (4096 vs
1024) because sub-call prompts include full data chunks.

### Process Reward Model (`reward_model.py`, 363 lines)

The reward model scores trajectory quality at each step. It has two
unique architectural features.

**A scalar reward head** (`reward_head.py`) projects the backbone's last
hidden state to a [0, 1] score:

```python
class RewardHead(nnx.Module):
    def __init__(self, hidden_dim, *, rngs):
        self.linear = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, last_hidden_state):
        return jax.nn.sigmoid(self.linear(last_hidden_state)).squeeze(-1)
```

The hidden dimension is auto-detected from the model config (896 for
Qwen2.5-0.5B, 1536 for 1.5B). This avoids hardcoding model-specific
values.

**Differential learning rates** via `optax.multi_transform`:

```python
optimizer = optax.multi_transform(
    transforms={
        "backbone": optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(backbone_schedule, weight_decay=0.01),  # 2e-5
        ),
        "head": optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(head_schedule, weight_decay=0.0),       # 1e-4
        ),
    },
    param_labels=label_tree,
)
```

The backbone (LoRA params on frozen base) trains at 2e-5. The head
(initialized randomly) trains at 1e-4 — five times faster, because it
needs stronger gradient signal to learn from scratch. The backbone already
understands language; the head needs to learn what "good" means.

Labels include **step-level penalties**: a correct trajectory scores 1.0
at normal steps, but only 0.3 at error steps and 0.1 at steps that
produced empty output. This teaches the model that errors and wasted
iterations are signs of a lower-quality trajectory, even if the final
answer is correct.

---

## 7. Serving: Closing the Loop

`serve.py` (352 lines) provides a multi-model OpenAI-compatible HTTP
server that routes requests by model name. It's designed to be a drop-in
replacement for the API endpoints that `micro_rlm.py` calls.

### Loading Trained Models

The `LoadedModel` class reconstructs a trained model from its saved
artifacts:

1. Load `train_config.json` for hyperparameters (base model, LoRA rank/alpha)
2. Load the base model via EasyDel
3. Inject LoRA structure (same `inject_lora` as training)
4. Restore LoRA weights from the Orbax checkpoint
5. Merge back into a single model for fast inference
6. Warm up JIT with a dummy forward pass

This means the server loads the full base model plus the saved LoRA
weights (~20 MB), not a separate copy of the full model. Multiple LoRA
specializations (root, sub, reward) can share the same base model in
memory, though the current implementation loads each independently.

### Bucket-Based JIT

JAX's JIT compiler produces a separate compiled program for each unique
input shape. Naively, every different prompt length would trigger a
recompilation. The server uses static buckets to avoid this:

```python
BUCKETS = [256, 512, 1024, 2048, 4096]

def find_bucket(length):
    for b in BUCKETS:
        if b >= length:
            return b
    return length
```

Inputs are padded to the next bucket boundary, with the attention mask
zeroing out padding positions. This limits recompilation to at most five
programs for typical workloads.

### The Virtuous Cycle

With the server running, the inference engine points at `localhost`
instead of a cloud API:

```
Phase 0:  micro_rlm.py --log ./traj          → generate trajectories (API models)
Phase 1:  reward_model.py --data ./traj       → train trajectory verifier
Phase 2:  sft_root.py --data ./traj           → train root model
          sft_sub.py --data ./traj            → train sub-call specialist
Phase 3:  serve.py --root-model ... --sub-model ...
          micro_rlm.py --base_url localhost   → generate with local models
Phase 4:  retrain on expanded data → repeat
```

After Phase 3, all inference is local — zero API cost. The locally
generated trajectories become training data for the next cycle. The paper
reports diminishing returns after roughly 1,000 high-quality trajectories;
2–3 cycles should suffice for most tasks.

---

## 8. Making It Fit: Consumer GPU Engineering

The entire pipeline runs on a single NVIDIA RTX 4070 Ti (12 GB VRAM).
Getting there required solving six distinct failure modes in the
JAX/EasyDel stack — the full debugging story is documented separately in
[`train/TRAINING_ON_CONSUMER_GPU.md`](../train/TRAINING_ON_CONSUMER_GPU.md).
Here we summarize the three highest-impact fixes and the memory reality.

### The Memory Budget

On our 12 GB card under WSL2, the baseline (driver + display server) eats
~3.3 GB, leaving ~8.9 GB for ML workloads:

| Model | Weights (bf16) | Training VRAM | Fits 12 GB? |
|---|---|---|---|
| Qwen2.5-0.5B | 1.3 GB | ~5 GB | Yes (3.9 GB free) |
| Qwen2.5-1.5B | 3.6 GB | ~9 GB at seq 512 | Barely (seq <= 512 only) |
| Qwen2.5-3B | 6.6 GB | N/A | No (OOMs on load) |

The 1.5B model fits at very short sequences, but the RLM system prompt
alone consumes ~600 tokens, leaving almost no room for actual trajectory
content. The 0.5B model at seq_len=1024 is the practical choice.

### Three Fixes That Mattered Most

**1. Disable the precomputed causal mask.** EasyDel precomputes a causal
attention mask at the model's maximum position embedding length. For
Qwen2.5 (max 32,768 positions), this is a 32K x 32K boolean tensor —
**1.07 GB** allocated on the GPU at model initialization, regardless of
actual sequence length. One line eliminates it:

```python
model.config.precompute_masks = False
```

Masks are then computed on-the-fly for the actual sequence length. At
seq_len=1024, the mask is 1 MB instead of 1 GB.

**2. Pass base_state as a JIT argument.** When frozen model weights
(3.5 GB for 1.5B) are captured in a `@jax.jit` closure, XLA embeds them
as MLIR constants, effectively doubling memory usage during compilation.
Moving `base_state` from closure capture to an explicit function argument
eliminates this.

**3. Use numpy for data shuffling.** `jax.random.permutation().tolist()`
allocates a JAX array on the GPU, then copies it to CPU. When GPU memory
is nearly full, even this tiny allocation can OOM. Using
`np.random.default_rng(seed).permutation()` runs entirely on the CPU.

### The Dependency Maze

The JAX ecosystem requires tight version coordination. Python 3.10 caps
JAX at 0.6.2 (0.7+ requires Python 3.11). Orbax 0.7.0 references a JAX
internal API removed in 0.6. Chex 0.1.91 requires JAX 0.7.0. EasyDel
0.1.4.1 is pinned because the API changes between every release.

The verified working set:

```
jax[cuda12]==0.6.2    flax==0.10.4       optax==0.2.5
orbax-checkpoint==0.11.32   chex==0.1.90   easydel==0.1.4.1
transformers==4.57.6   numpy<2.3
```

Every version is individually tested. See [`train/requirements.txt`](../train/requirements.txt)
for the pinned constraints.

---

## 9. What Makes This Different

Most fine-tuning projects train a model to answer questions better or
follow instructions more accurately. This system trains a model to
*interact with a code execution environment in a specific way*. Four
aspects make it unusual.

**Self-generated training data.** The trajectories used for training are
produced by the inference system itself. An API model (Claude, GPT-5)
runs the RLM loop, its interactions are logged, correct trajectories are
filtered, and the local model is trained on those interactions. There is
no human annotation — the ground truth comes from deterministic task
generators with verifiable answers.

**Behavioral pattern learning.** The model isn't learning Q&A pairs. It's
learning an interaction protocol: peek at context structure, design a
chunking strategy, use `llm_query()` on chunks inside loops, store
intermediate results in variables, aggregate, and emit `FINAL`. After
training, the model exhibits this behavior on new tasks without explicit
programming — it has internalized the RLM paradigm.

**Three complementary models.** The root model (the RLM controller), the
sub-model (the worker), and the reward model (the judge) each serve a
distinct role. They share the same base model and LoRA infrastructure but
are trained on different data for different objectives. At inference time,
they collaborate: the root writes code, the sub executes semantic
sub-tasks, and the reward model (optionally) scores trajectory quality.

**Consumer hardware.** Everything demonstrated here runs on a single 12 GB
GPU — the kind of card that costs $400 on the used market. No multi-GPU
setups, no cloud instances, no quantization tricks. The constraints forced
us to find and fix real memory bugs in the stack, which we've documented
for the community.

### Limitations

This is an educational implementation, not a production system:

- Model sizes are small (0.5B) due to hardware constraints
- Training data is synthetic (from the built-in demo tasks)
- The reward-filtered training loop isn't wired end-to-end yet
- No async sub-calls (the paper notes this as a key optimization)
- The REPL is unsandboxed (`exec()` on LLM-generated code)

---

## 10. Codebase Map

```
micro-rlm/
├── micro_rlm.py              535 lines   RLM inference engine (zero deps)
├── CLAUDE.md                              Developer guide
├── README.md                              Project overview & quick start
├── docs/
│   ├── ARCHITECTURE.md                    This document
│   └── 2512.24601v2.pdf                   The RLM paper
├── train/
│   ├── README.md                          Training pipeline overview
│   ├── TRAINING_ON_CONSUMER_GPU.md        GPU debugging guide (6 bugs)
│   ├── requirements.txt        23 lines   Pinned JAX ecosystem deps
│   ├── data_utils.py          347 lines   Trajectory → training data
│   ├── reward_head.py          34 lines   Scalar reward head (NNX)
│   ├── sft_root.py            402 lines   Root model LoRA SFT
│   ├── sft_sub.py             279 lines   Sub-call specialist SFT
│   ├── reward_model.py        363 lines   Process reward model
│   └── serve.py               352 lines   Multi-model HTTP server
│   └── traj_synthetic/
│       └── traj_*.json         12 files   Example trajectory data
```

**Total: 2,335 lines** of Python across the inference engine and training
pipeline.

**Dependency graph:**

```
micro_rlm.py  ← (stdlib only, zero external deps)
     ↑
     │ imports task generators + system prompt
     │
data_utils.py ← (pure Python + transformers tokenizer)
     ↑
     │ imports LoRA, inject_lora, make_mesh, make_cosine_schedule
     │
sft_root.py   ← (JAX, Flax NNX, EasyDel, Optax, Orbax)
     ↑                ↑               ↑
     │                │               │
sft_sub.py    reward_model.py    serve.py
                      ↑
                      │
                reward_head.py
```

The zero-dependency boundary at `micro_rlm.py` is intentional — the
inference engine runs anywhere with Python 3.8+ and an API key. The
training pipeline adds JAX and its ecosystem, but only in the `train/`
directory.

---

## 11. Where It Goes From Here

The 2,335 lines across this codebase capture a complete ML system:
inference, trajectory logging, data processing, three training scripts,
a reward model, and a multi-model serving layer. It demonstrates that the
RLM paradigm from the paper can be implemented, trained, and served on
hardware that most ML engineers already own.

The paper points toward several extensions that would build on this
foundation:

- **Async sub-calls** — all sub-calls in the current implementation are
  sequential. The paper notes that batching sub-calls (e.g., processing
  all chunks in parallel) would reduce latency proportionally.
- **Deeper recursion** — sub-calls could themselves be RLMs. A root RLM
  decomposes a problem; each sub-RLM processes its slice with its own
  REPL and sub-sub-calls. The paper explores depth-2 recursion for
  O(N^2) tasks.
- **Online reward search** — the reward model could guide beam search
  over RLM trajectories, pruning unproductive branches in real time
  rather than filtering after the fact.
- **Integration with RLHF/DPO** — the reward model and trajectory data
  provide the components for reinforcement learning from human feedback
  or direct preference optimization, going beyond pure SFT.

The codebase is designed for learning and experimentation. Every function
is documented, every design decision is explained, and the training
pipeline runs end-to-end in under 10 minutes. If you're interested in
RLMs, clone the repo and start generating trajectories.

---

*Built as part of [micro-rlm](https://github.com/Sha01in/micro-rlm), a
minimal implementation of [Recursive Language Models](https://arxiv.org/abs/2512.24601)
(Zhang, Kraska, Khattab — MIT CSAIL, 2026).*
