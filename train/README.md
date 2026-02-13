# micro-rlm Training Pipeline

JAX-based training for fine-tuning local models on RLM trajectories.
Closes the loop: generate trajectories → score/filter → fine-tune → serve → generate better trajectories.

All training fits on a single RTX 4070Ti (12GB VRAM). For the full
technical narrative, see [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).
For the detailed GPU debugging guide, see
[`TRAINING_ON_CONSUMER_GPU.md`](TRAINING_ON_CONSUMER_GPU.md).

## Setup

```bash
# Requires CUDA 12+ and compatible NVIDIA driver
pip install -r train/requirements.txt

# Verify JAX GPU access
python -c "import jax; print(jax.devices())"
```

**Environment variables** (recommended for 12 GB cards):
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
```

## The Virtuous Cycle

```
Phase 0: Generate trajectories with API models
  python micro_rlm.py --task census --n_entries 200 --log ./traj
  python micro_rlm.py --task census --n_entries 500 --seed 123 --log ./traj

Phase 1: Train verifier on initial trajectories
  python train/reward_model.py --data ./traj --output ./models/reward_v1

Phase 2: Train root + sub models on reward-filtered data
  python train/sft_root.py --data ./traj --reward_model ./models/reward_v1 \
      --reward_threshold 0.7 --output ./models/root_v1
  python train/sft_sub.py --data ./traj --reward_model ./models/reward_v1 \
      --reward_threshold 0.7 --output ./models/sub_v1

Phase 3: Serve locally and generate better trajectories
  python train/serve.py --root-model ./models/root_v1 --sub-model ./models/sub_v1
  python micro_rlm.py --model root --sub_model sub \
      --base_url http://localhost:8000/v1 --api_key dummy --log ./traj_v2

Phase 4: Retrain on expanded data → repeat
```

The paper reports diminishing returns after ~1000 high-quality trajectories. 2-3 cycles should suffice.

## Enhancement 1: Root Model SFT

Teaches a local model the RLM interaction pattern (write REPL code, use `llm_query()`, emit FINAL).

```bash
python train/sft_root.py --data ./trajectories --output ./models/root_v1
```

| Setting | Default | Notes |
|---------|---------|-------|
| Base model | Qwen2.5-0.5B-Instruct | ~1.3 GB in bf16 |
| LoRA rank | 16 | alpha=32, all linear layers (168 layers) |
| Learning rate | 2e-4 | Cosine decay with warmup |
| Epochs | 3 | |
| Seq length | 1024 | Multi-turn RLM conversations |
| Batch | 1 × 8 grad_accum | Effective batch size 8 |
| VRAM | ~5 GB | With gradient checkpointing |

## Enhancement 2: Sub-Call Specialist

Eliminates sub-call API costs with a fine-tuned local model. Sub-calls are simple stateless tasks that a small model handles well.

```bash
python train/sft_sub.py --data ./trajectories --output ./models/sub_v1
```

| Setting | Default | Notes |
|---------|---------|-------|
| Base model | Qwen2.5-0.5B-Instruct | ~1.3 GB in bf16 |
| LoRA rank | 16 | alpha=32 |
| Learning rate | 5e-5 | |
| Epochs | 5 | More passes over simpler data |
| Seq length | 4096 | Sub-calls include full data chunks |
| Batch | 2 × 4 grad_accum | |
| VRAM | ~5 GB | |

## Enhancement 3: Trajectory Verifier

Process reward model that scores each step of an RLM trajectory. Uses differential learning rates: the reward head (trained from scratch) gets 5x higher LR than the LoRA backbone.

```bash
python train/reward_model.py --data ./trajectories --output ./models/reward_v1
```

| Setting | Default | Notes |
|---------|---------|-------|
| Base model | Qwen2.5-0.5B-Instruct | Hidden dim auto-detected |
| LoRA rank | 16 | On backbone |
| Backbone LR | 2e-5 | Fine-tuning |
| Head LR | 1e-4 | Training from scratch |
| Epochs | 10 | Small dataset benefits from more passes |
| Loss | BCE | Or MSE with `--loss_type mse` |
| Label smoothing | 0.05 | |
| VRAM | ~5 GB | |

## Serving

Multi-model OpenAI-compatible server. Routes requests by the `model` field.

```bash
# Root only
python train/serve.py --root-model ./models/root_v1

# Root + sub
python train/serve.py --root-model ./models/root_v1 --sub-model ./models/sub_v1

# All three
python train/serve.py --root-model ./models/root_v1 --sub-model ./models/sub_v1 \
    --reward-model ./models/reward_v1
```

Endpoint: `POST /v1/chat/completions` (OpenAI-compatible).

## VRAM Budget (RTX 4070Ti, 12GB)

| Phase | What's in VRAM | Est. Usage |
|-------|---------------|------------|
| Train root SFT | Qwen2.5-0.5B (bf16) + LoRA + optimizer | ~5 GB |
| Train sub SFT | Qwen2.5-0.5B (bf16) + LoRA + optimizer | ~5 GB |
| Train reward | Qwen2.5-0.5B (bf16) + LoRA + head | ~5 GB |
| Serve root + sub | 2× Qwen2.5-0.5B (bf16) | ~6 GB |
| Serve all three | 3× Qwen2.5-0.5B (bf16) | ~8 GB |

Scaling to larger GPUs: 16 GB fits Qwen2.5-1.5B, 24 GB fits Qwen2.5-3B.

## File Structure

```
train/
├── README.md                          ← this file
├── TRAINING_ON_CONSUMER_GPU.md        ← GPU debugging guide (6 bugs fixed)
├── requirements.txt                   ← JAX 0.6.2, EasyDel 0.1.4.1, etc.
├── data_utils.py                      ← trajectory loading, filtering, ChatML, tokenization
├── reward_head.py                     ← Flax NNX reward head module (~34 lines)
├── sft_root.py                        ← Enhancement 1: root model LoRA SFT
├── sft_sub.py                         ← Enhancement 2: sub-call specialist LoRA SFT
├── reward_model.py                    ← Enhancement 3: process reward model
└── serve.py                           ← multi-model OpenAI-compatible server
```

## Notes

- **XLA compilation warmup**: The first training step or inference call triggers XLA compilation (~60s for 0.5B models). This is a one-time cost per session — the program is not hanging.
- **EasyDel version**: Pinned to `==0.1.4.1` in requirements.txt. The API changes between releases — do not upgrade without testing. See [EasyDel releases](https://github.com/erfanzar/EasyDeL/releases).
- **Chat template**: The server applies the same Qwen ChatML template used during training via `tokenizer.apply_chat_template()`. Using a different tokenizer/template will degrade quality.
- **micro_rlm.py untouched**: All training code lives in `train/`. The main script remains zero-dependency.
- **GPU memory**: If you encounter OOM errors, see [TRAINING_ON_CONSUMER_GPU.md](TRAINING_ON_CONSUMER_GPU.md) for the full debugging guide.
