"""Sub-call specialist LoRA SFT — eliminate sub-call API costs.

Fine-tunes Qwen2.5-0.5B-Instruct on sub-call (prompt, response) pairs
extracted from successful RLM trajectories. Sub-calls are simple stateless
tasks (analyze chunk, count items, extract info) that a small fine-tuned
model handles well.

This implements Enhancement 2 from the training plan.

Usage:
    python train/sft_sub.py --data ./trajectories --output ./models/sub_v1
    python train/sft_sub.py --data ./traj --reward_model ./models/reward_v1 \\
        --reward_threshold 0.7 --output ./models/sub_v1

Environment variables for GPU memory management:
    XLA_PYTHON_CLIENT_PREALLOCATE=false   — allocate GPU memory on demand
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   — use up to 95% of GPU memory

VRAM: ~5-6 GB on RTX 4070Ti (12GB).
Note: First training step triggers XLA compilation (~1-2 min).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrandom
import optax
import orbax.checkpoint as ocp
from flax import nnx
from transformers import AutoTokenizer
from easydel import AutoEasyDeLModelForCausalLM

from data_utils import (
    load_trajectories,
    filter_correct,
    extract_sub_calls,
    tokenize_chat,
    pad_and_batch,
    print_data_summary,
)
from sft_root import (
    LoRAParam,
    LoRALinear,
    TARGET_MODULES,
    inject_lora,
    count_params,
    make_cosine_schedule,
    make_mesh,
)


# ── CLI ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Sub-call specialist LoRA SFT")
    # Data
    p.add_argument("--data", required=True, help="Directory with trajectory JSONs")
    p.add_argument("--output", required=True, help="Output directory for LoRA weights")
    p.add_argument("--reward_model", default=None,
                   help="Path to reward model for data filtering")
    p.add_argument("--reward_threshold", type=float, default=0.7)
    # Model — 0.5B is small enough for full-precision LoRA on 12GB
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace model ID for the base model")
    # LoRA — smaller rank for smaller model
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    # Training — different hyperparams than root
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seq_len", type=int, default=4096,
                   help="Longer than root — sub-calls include full data chunks")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = jrandom.PRNGKey(args.seed)

    # ── 1. Load trajectories and extract sub-calls ──────────────
    print(f"Loading trajectories from {args.data}")
    trajs = load_trajectories(args.data, mode="rlm")
    print(f"  Loaded {len(trajs)} RLM trajectories")

    trajs = filter_correct(trajs)
    print(f"  After correctness filter: {len(trajs)}")
    print_data_summary(trajs, "filtered")

    if not trajs:
        print("No valid trajectories. Generate some first:")
        print("  python micro_rlm.py --task census --n_entries 200 --log ./trajectories")
        sys.exit(1)

    if args.reward_model:
        print(f"  Reward filtering (threshold={args.reward_threshold}): "
              f"pass --reward_model to enable (not yet wired)")

    pairs = extract_sub_calls(trajs)
    print(f"\n  Extracted {len(pairs)} sub-call pairs")
    if pairs:
        avg_prompt = sum(len(p["messages"][0]["content"]) for p in pairs) / len(pairs)
        avg_resp = sum(len(p["messages"][1]["content"]) for p in pairs) / len(pairs)
        print(f"  Avg prompt: {avg_prompt:.0f} chars, avg response: {avg_resp:.0f} chars")

    if not pairs:
        print("No sub-calls found. Ensure trajectories contain sub-call data.")
        sys.exit(1)

    # ── 2. Tokenize ─────────────────────────────────────────────
    print(f"\nTokenizing with {args.base_model} tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = []
    for pair in pairs:
        tok = tokenize_chat(pair["messages"], tokenizer, args.seq_len, mask_user=True)
        if tok:
            samples.append(tok)

    print(f"  {len(samples)} tokenized samples (max_len={args.seq_len})")
    if samples:
        avg_len = sum(len(s["input_ids"]) for s in samples) / len(samples)
        print(f"  Avg sequence length: {avg_len:.0f} tokens")

    if not samples:
        print("No valid tokenized samples.")
        sys.exit(1)

    # ── 3. Load model ───────────────────────────────────────────
    print(f"\nLoading model: {args.base_model}")

    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        auto_shard_model=False,
    )

    # Disable precomputed causal mask to save ~1 GB GPU memory
    model.config.precompute_masks = False

    # ── 4. Inject LoRA + split ──────────────────────────────────
    rng, lora_rng = jrandom.split(rng)
    n_injected = inject_lora(model, args.lora_rank, args.lora_alpha, lora_rng)

    graphdef, lora_state, base_state = nnx.split(model, LoRAParam, ...)
    del model
    n_lora = count_params(lora_state)
    n_base = count_params(base_state)
    print(f"\n  Base params: {n_base:,}")
    print(f"  LoRA params: {n_lora:,} (rank={args.lora_rank}, alpha={args.lora_alpha})")
    print(f"  LoRA layers: {n_injected}")

    # ── 5. Optimizer ────────────────────────────────────────────
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(samples) / effective_batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    schedule = make_cosine_schedule(args.lr, warmup_steps, total_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    if args.grad_accum > 1:
        optimizer = optax.MultiSteps(optimizer, args.grad_accum)

    opt_state = optimizer.init(lora_state)

    # ── 6. JIT-compiled training step ───────────────────────────
    @jax.jit
    def train_step(lora_state, opt_state, base_state, batch):
        def loss_fn(ls):
            model = nnx.merge(graphdef, ls, base_state)
            output = jax.checkpoint(
                lambda ids, mask: model(input_ids=ids, attention_mask=mask),
            )(batch["input_ids"], batch["attention_mask"])
            logits = output.logits

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

    # ── 7. Training loop ────────────────────────────────────────
    print(f"\nTraining: {args.epochs} epochs, {total_steps} optimizer steps")
    print(f"  Batch: {args.batch_size} x {args.grad_accum} = {effective_batch}")
    print(f"  LR: {args.lr} (cosine, {warmup_steps} warmup)")

    mesh = make_mesh()

    t0_total = time.time()
    global_step = 0
    best_loss = float("inf")
    np_rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        t0_epoch = time.time()
        indices = np_rng.permutation(len(samples)).tolist()
        epoch_samples = [samples[i] for i in indices]
        batches = pad_and_batch(epoch_samples, args.batch_size, tokenizer.pad_token_id)
        epoch_loss = 0.0

        for step, batch in enumerate(batches):
            jax_batch = {k: jnp.array(v, dtype=jnp.int32) for k, v in batch.items()}
            with mesh:
                lora_state, opt_state, loss = train_step(
                    lora_state, opt_state, base_state, jax_batch,
                )
            loss_val = float(loss)
            epoch_loss += loss_val
            global_step += 1

            if step == 0 and epoch == 0:
                print(f"  XLA compilation took {time.time() - t0_total:.1f}s")

            if global_step % 10 == 0 or step == len(batches) - 1:
                avg = epoch_loss / (step + 1)
                print(f"  Epoch {epoch+1}/{args.epochs}  "
                      f"step {step+1}/{len(batches)}  "
                      f"loss={loss_val:.4f}  avg={avg:.4f}")

        avg_loss = epoch_loss / max(len(batches), 1)
        print(f"  Epoch {epoch+1} complete — avg_loss={avg_loss:.4f} "
              f"({time.time() - t0_epoch:.1f}s)")
        if avg_loss < best_loss:
            best_loss = avg_loss

    total_time = time.time() - t0_total
    print(f"\nTraining complete: {total_time:.1f}s, best avg_loss={best_loss:.4f}")

    # ── 8. Save ─────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    checkpointer = ocp.PyTreeCheckpointer()
    lora_path = os.path.join(args.output, "lora_params")
    checkpointer.save(lora_path, lora_state)
    tokenizer.save_pretrained(args.output)

    config = {
        **vars(args),
        "n_sub_call_pairs": len(pairs),
        "n_samples": len(samples),
        "n_lora_params": n_lora,
        "best_loss": best_loss,
        "total_time_s": round(total_time, 1),
    }
    with open(os.path.join(args.output, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nModel saved to {args.output}")
    print(f"  Serve: python train/serve.py --sub-model {args.output}")


if __name__ == "__main__":
    main()
