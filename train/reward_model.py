"""Trajectory verifier — process reward model for scoring RLM trajectories.

Trains a reward model (Qwen2.5-0.5B backbone + scalar reward head) that
scores each step of an RLM trajectory. Used to:
  (a) filter SFT training data to high-quality trajectories
  (b) early-stop bad trajectories at inference time
  (c) best-of-N code selection

This implements Enhancement 3 from the training plan.

Usage:
    python train/reward_model.py --data ./trajectories --output ./models/reward_v1

Environment variables for GPU memory management:
    XLA_PYTHON_CLIENT_PREALLOCATE=false   — allocate GPU memory on demand
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   — use up to 95% of GPU memory

VRAM: ~5-6 GB on RTX 4070Ti (12GB).
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
    label_trajectories,
    print_data_summary,
)
from sft_root import (
    LoRAParam,
    inject_lora,
    count_params,
    make_cosine_schedule,
    make_mesh,
)
from reward_head import RewardHead


# ── CLI ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Process reward model for micro-rlm")
    # Data
    p.add_argument("--data", required=True, help="Directory with trajectory JSONs")
    p.add_argument("--output", required=True, help="Output directory")
    # Model
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Backbone model")
    # LoRA on backbone
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    # Training
    p.add_argument("--backbone_lr", type=float, default=2e-5,
                   help="Learning rate for LoRA backbone params")
    p.add_argument("--head_lr", type=float, default=1e-4,
                   help="Learning rate for reward head (trained from scratch)")
    p.add_argument("--epochs", type=int, default=10,
                   help="More epochs for small reward datasets")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.05,
                   help="Label smoothing for BCE loss")
    p.add_argument("--loss_type", default="bce", choices=["bce", "mse"],
                   help="bce for outcome supervision, mse for process supervision")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Reward-specific data processing ─────────────────────────────

def tokenize_reward_samples(samples, tokenizer, max_len):
    """Tokenize reward model samples, recording last token position."""
    tokenized = []
    for sample in samples:
        text = tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer(text, truncation=True, max_length=max_len)
        input_ids = tokens["input_ids"]
        if len(input_ids) < 4:
            continue
        tokenized.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "reward_label": sample["label"],
            "last_pos": len(input_ids) - 1,
        })
    return tokenized


def pad_and_batch_reward(samples, batch_size, pad_id=0):
    """Pad and batch reward model samples."""
    batches = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        max_len = max(len(s["input_ids"]) for s in batch)

        input_ids, masks, labels, positions = [], [], [], []
        for s in batch:
            pad_len = max_len - len(s["input_ids"])
            input_ids.append(s["input_ids"] + [pad_id] * pad_len)
            masks.append(s["attention_mask"] + [0] * pad_len)
            labels.append(s["reward_label"])
            positions.append(s["last_pos"])

        batches.append({
            "input_ids": input_ids,
            "attention_mask": masks,
            "reward_labels": labels,
            "last_positions": positions,
        })
    return batches


# ── Reward head variable marker ─────────────────────────────────

class HeadParam(nnx.Variable):
    """Marker for reward head parameters (separate LR from LoRA)."""
    pass


def main():
    args = parse_args()
    rng = jrandom.PRNGKey(args.seed)

    # ── 1. Load and label trajectories ──────────────────────────
    print(f"Loading trajectories from {args.data}")
    trajs = load_trajectories(args.data, mode="rlm")
    print(f"  Loaded {len(trajs)} RLM trajectories")
    print_data_summary(trajs, "all")

    samples = label_trajectories(trajs)
    print(f"\n  Labeled {len(samples)} step-level samples")
    if samples:
        pos = sum(1 for s in samples if s["label"] > 0.5)
        neg = len(samples) - pos
        print(f"  Positive (>0.5): {pos}, Negative: {neg}")

    if not samples:
        print("No labeled samples. Generate trajectories first.")
        sys.exit(1)

    # ── 2. Tokenize ─────────────────────────────────────────────
    print(f"\nTokenizing with {args.base_model} tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tok_samples = tokenize_reward_samples(samples, tokenizer, args.seq_len)
    print(f"  {len(tok_samples)} tokenized samples")

    if not tok_samples:
        print("No valid tokenized samples.")
        sys.exit(1)

    # ── 3. Load backbone model ──────────────────────────────────
    print(f"\nLoading backbone: {args.base_model}")
    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        auto_shard_model=False,
    )

    # Disable precomputed causal mask to save ~1 GB GPU memory
    model.config.precompute_masks = False

    # ── 4. Inject LoRA + create reward head ─────────────────────
    rng, lora_rng, head_rng = jrandom.split(rng, 3)

    n_injected = inject_lora(model, args.lora_rank, args.lora_alpha, lora_rng)
    print(f"  LoRA layers injected: {n_injected}")

    # Create reward head — detect hidden_dim from model config
    hidden_dim = getattr(model.config, "hidden_size", 896)
    reward_head = RewardHead(hidden_dim, rngs=nnx.Rngs(head_rng))

    # Split backbone into LoRA (trainable) + base (frozen)
    graphdef, lora_state, base_state = nnx.split(model, LoRAParam, ...)
    del model
    n_lora = count_params(lora_state)
    print(f"  LoRA params: {n_lora:,} (rank={args.lora_rank})")

    # Split reward head for functional training
    head_graphdef, head_state = nnx.split(reward_head)
    n_head = count_params(head_state)
    print(f"  Reward head params: {n_head:,}")

    # ── 5. Differential LR optimizer via optax.multi_transform ──
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(tok_samples) / effective_batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    backbone_schedule = make_cosine_schedule(args.backbone_lr, warmup_steps, total_steps)
    head_schedule = make_cosine_schedule(args.head_lr, warmup_steps, total_steps)

    # Group params with labels for multi_transform
    all_params = {"lora": lora_state, "head": head_state}

    lora_labels = jax.tree.map(lambda _: "backbone", lora_state)
    head_labels = jax.tree.map(lambda _: "head", head_state)
    label_tree = {"lora": lora_labels, "head": head_labels}

    optimizer = optax.multi_transform(
        transforms={
            "backbone": optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adamw(backbone_schedule, weight_decay=0.01),
            ),
            "head": optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adamw(head_schedule, weight_decay=0.0),
            ),
        },
        param_labels=label_tree,
    )
    if args.grad_accum > 1:
        optimizer = optax.MultiSteps(optimizer, args.grad_accum)

    opt_state = optimizer.init(all_params)

    # ── 6. JIT-compiled training step ───────────────────────────
    smooth = args.label_smoothing

    @jax.jit
    def train_step(all_params, opt_state, base_state, batch):
        def loss_fn(ap):
            # Reconstruct backbone with LoRA
            backbone = nnx.merge(graphdef, ap["lora"], base_state)

            # Forward pass — get hidden states with gradient checkpointing
            output = jax.checkpoint(
                lambda ids, mask: backbone(
                    input_ids=ids, attention_mask=mask,
                    output_hidden_states=True,
                ),
            )(batch["input_ids"], batch["attention_mask"])

            # Extract last hidden state at last-token position per sample
            hidden = output.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            last_pos = batch["last_positions"]
            batch_idx = jnp.arange(hidden.shape[0])
            last_hidden = hidden[batch_idx, last_pos]  # (batch, hidden_dim)

            # Reward head
            head = nnx.merge(head_graphdef, ap["head"])
            scores = head(last_hidden)  # (batch,)

            # Loss
            targets = batch["reward_labels"]
            if smooth > 0:
                targets = targets * (1 - smooth) + 0.5 * smooth

            if args.loss_type == "bce":
                loss = optax.sigmoid_binary_cross_entropy(
                    jax.nn.logit(scores.clip(1e-7, 1 - 1e-7)), targets
                ).mean()
            else:  # mse
                loss = jnp.mean((scores - targets) ** 2)

            return loss

        loss, grads = jax.value_and_grad(loss_fn)(all_params)
        updates, new_opt_state = optimizer.update(grads, opt_state, all_params)
        new_params = optax.apply_updates(all_params, updates)
        return new_params, new_opt_state, loss

    # ── 7. Training loop ────────────────────────────────────────
    print(f"\nTraining: {args.epochs} epochs, {total_steps} optimizer steps")
    print(f"  Loss: {args.loss_type}, label_smoothing={args.label_smoothing}")
    print(f"  Backbone LR: {args.backbone_lr}, Head LR: {args.head_lr}")

    mesh = make_mesh()

    t0_total = time.time()
    global_step = 0
    best_loss = float("inf")
    np_rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        t0_epoch = time.time()
        indices = np_rng.permutation(len(tok_samples)).tolist()
        epoch_samples = [tok_samples[i] for i in indices]
        batches = pad_and_batch_reward(epoch_samples, args.batch_size, tokenizer.pad_token_id)
        epoch_loss = 0.0

        for step, batch in enumerate(batches):
            jax_batch = {
                "input_ids": jnp.array(batch["input_ids"], dtype=jnp.int32),
                "attention_mask": jnp.array(batch["attention_mask"], dtype=jnp.int32),
                "reward_labels": jnp.array(batch["reward_labels"], dtype=jnp.float32),
                "last_positions": jnp.array(batch["last_positions"], dtype=jnp.int32),
            }

            with mesh:
                all_params, opt_state, loss = train_step(
                    all_params, opt_state, base_state, jax_batch,
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
    checkpointer.save(os.path.join(args.output, "lora_params"), all_params["lora"])
    checkpointer.save(os.path.join(args.output, "head_params"), all_params["head"])
    tokenizer.save_pretrained(args.output)

    config = {
        **vars(args),
        "hidden_dim": hidden_dim,
        "n_samples": len(tok_samples),
        "n_lora_params": n_lora,
        "best_loss": best_loss,
        "total_time_s": round(total_time, 1),
    }
    with open(os.path.join(args.output, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nReward model saved to {args.output}")
    print(f"  Use with SFT: python train/sft_root.py --reward_model {args.output} ...")


if __name__ == "__main__":
    main()
