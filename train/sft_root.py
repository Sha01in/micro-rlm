"""Root model LoRA SFT — teach a local model to BE the RLM.

Fine-tunes Qwen2.5-0.5B-Instruct (default) on successful RLM trajectories using
LoRA via JAX/Flax NNX/EasyDel. The model learns the RLM interaction
pattern: write REPL code, use llm_query() on chunks, store results,
emit FINAL.

This implements Enhancement 1 from the training plan.

Usage:
    python train/sft_root.py --data ./trajectories --output ./models/root_v1
    python train/sft_root.py --data ./traj --reward_model ./models/reward_v1 \\
        --reward_threshold 0.7 --output ./models/root_v1

Environment variables for GPU memory management:
    XLA_PYTHON_CLIENT_PREALLOCATE=false   — allocate GPU memory on demand
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   — use up to 95% of GPU memory

VRAM: ~5 GB on RTX 4070Ti (12GB) with 0.5B model; ~9 GB with 1.5B.
Note: First training step triggers XLA compilation (~1-3 min). This is normal.
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
from jax.sharding import Mesh
import optax
import orbax.checkpoint as ocp
from flax import nnx
from transformers import AutoTokenizer
from easydel import AutoEasyDeLModelForCausalLM

from data_utils import (
    load_trajectories,
    filter_correct,
    traj_to_root_messages,
    tokenize_chat,
    pad_and_batch,
    print_data_summary,
)


# ── CLI ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Root model LoRA SFT for micro-rlm")
    # Data
    p.add_argument("--data", required=True, help="Directory with trajectory JSONs")
    p.add_argument("--output", required=True, help="Output directory for LoRA weights")
    p.add_argument("--reward_model", default=None,
                   help="Path to reward model for data filtering")
    p.add_argument("--reward_threshold", type=float, default=0.7,
                   help="Minimum reward score to keep a trajectory")
    # Model
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace model ID (0.5B fits 12GB; 1.5B needs 24GB)")
    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    # Training
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── LoRA via Flax NNX ──────────────────────────────────────────
#
# EasyDel 0.1.x returns Flax NNX modules. We use nnx.split/merge
# for functional training with jax.value_and_grad.
#
# LoRA approach: replace target Linear layers with LoRALinear modules
# that add low-rank A @ B contributions. Only LoRA params are trained.


class LoRAParam(nnx.Variable):
    """Marker variable type for trainable LoRA parameters.

    Lets us use nnx.split(model, LoRAParam, ...) to separate
    trainable LoRA params from frozen base params.
    """
    pass


class LoRALinear(nnx.Module):
    """Linear layer with LoRA adaptation.

    Wraps an existing nnx.Linear, adding a low-rank A @ B residual.
    Base weights are frozen; only A and B are optimized.

    LoRA: y = base(x) + (x @ A @ B) * (alpha / rank)
    """

    def __init__(self, base_linear, rank, alpha, *, rngs):
        self.base = base_linear
        self.rank = rank
        self.scale = alpha / rank
        kernel = base_linear.kernel.value
        in_features, out_features = kernel.shape
        key = rngs()
        self.lora_A = LoRAParam(
            jrandom.normal(key, (in_features, rank), dtype=jnp.bfloat16) / rank
        )
        self.lora_B = LoRAParam(
            jnp.zeros((rank, out_features), dtype=jnp.bfloat16)
        )

    def __call__(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.value) @ self.lora_B.value * self.scale
        return base_out + lora_out


TARGET_MODULES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP
})


def _is_linear(layer):
    """Check if a layer is a linear module (nnx.Linear or EasyDel ParallelLinear)."""
    if isinstance(layer, nnx.Linear):
        return True
    # EasyDel uses ParallelLinear which has .kernel but isn't nnx.Linear
    return isinstance(layer, nnx.Module) and hasattr(layer, 'kernel')


def inject_lora(model, rank, alpha, rng):
    """Replace target Linear layers in the model with LoRALinear.

    Walks the model's module tree and swaps matching linear layers
    with LoRALinear wrappers. Supports both nnx.Linear and EasyDel's
    ParallelLinear. Returns the count of injected layers.
    """
    count = 0
    rngs = nnx.Rngs(rng)
    for path, module in model.iter_modules():
        for attr_name in list(vars(module)):
            if attr_name not in TARGET_MODULES:
                continue
            layer = getattr(module, attr_name, None)
            if _is_linear(layer):
                setattr(module, attr_name, LoRALinear(layer, rank, alpha, rngs=rngs))
                count += 1
    return count


def count_params(state):
    """Count parameters in a state pytree."""
    return sum(p.size for p in jax.tree.leaves(state))


def make_mesh():
    """Create a single-device JAX mesh (required by EasyDel's sharding layer)."""
    return Mesh(np.array(jax.devices()), ('dp',))


# ── Training ────────────────────────────────────────────────────

def make_cosine_schedule(peak_lr, warmup_steps, total_steps):
    """Cosine decay with linear warmup."""
    warmup = optax.linear_schedule(0.0, peak_lr, warmup_steps)
    decay = optax.cosine_decay_schedule(peak_lr, total_steps - warmup_steps)
    return optax.join_schedules([warmup, decay], [warmup_steps])


def main():
    args = parse_args()
    rng = jrandom.PRNGKey(args.seed)

    # ── 1. Load and filter trajectories ─────────────────────────
    print(f"Loading trajectories from {args.data}")
    trajs = load_trajectories(args.data, mode="rlm")
    print(f"  Loaded {len(trajs)} RLM trajectories")
    print_data_summary(trajs, "raw")

    trajs = filter_correct(trajs)
    print(f"  After correctness filter: {len(trajs)}")

    if not trajs:
        print("No valid trajectories. Generate some first:")
        print("  python micro_rlm.py --task census --n_entries 200 --log ./trajectories")
        sys.exit(1)

    if args.reward_model:
        print(f"  Reward filtering (threshold={args.reward_threshold}): "
              f"pass --reward_model to enable (not yet wired)")

    # ── 2. Tokenize ─────────────────────────────────────────────
    print(f"\nTokenizing with {args.base_model} tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = []
    for traj in trajs:
        messages = traj_to_root_messages(traj)
        if not messages:
            continue
        tok = tokenize_chat(messages, tokenizer, args.seq_len, mask_user=True)
        if tok:
            samples.append(tok)

    print(f"  {len(samples)} tokenized samples (max_len={args.seq_len})")
    if samples:
        avg_len = sum(len(s["input_ids"]) for s in samples) / len(samples)
        n_labeled = sum(sum(1 for l in s["labels"] if l != -100) for s in samples)
        print(f"  Avg sequence length: {avg_len:.0f} tokens")
        print(f"  Total supervised tokens: {n_labeled:,}")

    if not samples:
        print("No valid tokenized samples. Check trajectory format.")
        sys.exit(1)

    # ── 3. Load model (EasyDel 0.1.x returns a Flax NNX module) ─
    print(f"\nLoading model: {args.base_model}")
    print("  First-time download/compilation may take several minutes...")

    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        auto_shard_model=False,
    )

    # Disable precomputed causal mask to save ~1 GB GPU memory.
    # EasyDel precomputes a (max_pos x max_pos) mask on GPU by default;
    # with precompute_masks=False, masks are computed on-the-fly for the
    # actual sequence length.
    model.config.precompute_masks = False

    # ── 4. Inject LoRA layers ───────────────────────────────────
    rng, lora_rng = jrandom.split(rng)
    n_injected = inject_lora(model, args.lora_rank, args.lora_alpha, lora_rng)

    # Split into graph structure + trainable LoRA params + frozen base params.
    # Delete the original model reference to free any non-shared buffers.
    graphdef, lora_state, base_state = nnx.split(model, LoRAParam, ...)
    del model
    n_lora = count_params(lora_state)
    n_base = count_params(base_state)
    print(f"\n  Base params: {n_base:,} ({n_base * 2 / 1e9:.1f} GB in bf16)")
    print(f"  LoRA params: {n_lora:,} (rank={args.lora_rank}, alpha={args.lora_alpha})")
    print(f"  LoRA layers injected: {n_injected}")
    print(f"  Trainable: {n_lora / max(n_base, 1) * 100:.2f}%")

    # ── 5. Optimizer (only for LoRA params) ─────────────────────
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
    # The entire forward + backward compiles to a single XLA graph.
    # base_state is passed as an explicit argument (not captured in
    # the closure) to avoid XLA embedding it as multi-GB MLIR constants.

    @jax.jit
    def train_step(lora_state, opt_state, base_state, batch):
        def loss_fn(ls):
            # Reconstruct model from graph + LoRA params + frozen base
            model = nnx.merge(graphdef, ls, base_state)

            # Forward pass with gradient checkpointing to save memory.
            # The model is captured via closure inside the checkpoint;
            # only JAX arrays (ids, mask) are passed as arguments.
            output = jax.checkpoint(
                lambda ids, mask: model(input_ids=ids, attention_mask=mask),
            )(batch["input_ids"], batch["attention_mask"])
            logits = output.logits

            # Next-token prediction loss with label masking
            shift_logits = logits[:, :-1]
            shift_labels = batch["labels"][:, 1:]

            log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
            token_losses = -jnp.take_along_axis(
                log_probs, shift_labels[:, :, None], axis=-1
            ).squeeze(-1)

            mask = (shift_labels != -100).astype(jnp.float32)
            loss = (token_losses * mask).sum() / mask.sum().clip(min=1)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(lora_state)
        updates, new_opt_state = optimizer.update(grads, opt_state, lora_state)
        new_lora = optax.apply_updates(lora_state, updates)
        return new_lora, new_opt_state, loss

    # ── 7. Training loop ────────────────────────────────────────
    print(f"\nTraining: {args.epochs} epochs, {total_steps} optimizer steps")
    print(f"  Batch: {args.batch_size} x {args.grad_accum} grad_accum = {effective_batch}")
    print(f"  LR: {args.lr} (cosine, {warmup_steps} warmup steps)")
    print(f"  Seq len: {args.seq_len}")

    # EasyDel requires a JAX mesh context for sharding
    mesh = make_mesh()

    t0_total = time.time()
    global_step = 0
    best_loss = float("inf")
    np_rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        t0_epoch = time.time()

        # Shuffle samples each epoch (numpy to avoid GPU allocation)
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
                compile_time = time.time() - t0_total
                print(f"  XLA compilation took {compile_time:.1f}s")

            if global_step % 10 == 0 or step == len(batches) - 1:
                avg = epoch_loss / (step + 1)
                lr_now = float(schedule(global_step * args.grad_accum))
                print(f"  Epoch {epoch+1}/{args.epochs}  "
                      f"step {step+1}/{len(batches)}  "
                      f"loss={loss_val:.4f}  avg={avg:.4f}  lr={lr_now:.2e}")

        avg_loss = epoch_loss / max(len(batches), 1)
        elapsed = time.time() - t0_epoch
        print(f"  Epoch {epoch+1} complete — avg_loss={avg_loss:.4f} ({elapsed:.1f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss

    total_time = time.time() - t0_total
    print(f"\nTraining complete: {total_time:.1f}s total, best avg_loss={best_loss:.4f}")

    # ── 8. Save ─────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    # Save LoRA state with orbax
    checkpointer = ocp.PyTreeCheckpointer()
    lora_path = os.path.join(args.output, "lora_params")
    checkpointer.save(lora_path, lora_state)
    print(f"  Saved LoRA weights to {lora_path}")

    # Save tokenizer
    tokenizer.save_pretrained(args.output)
    print(f"  Saved tokenizer to {args.output}")

    # Save training config for reproducibility
    config = {
        **vars(args),
        "n_trajectories": len(trajs),
        "n_samples": len(samples),
        "n_lora_params": n_lora,
        "n_lora_layers": n_injected,
        "best_loss": best_loss,
        "total_time_s": round(total_time, 1),
    }
    with open(os.path.join(args.output, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nModel saved to {args.output}")
    print(f"  Serve: python train/serve.py --root-model {args.output}")


if __name__ == "__main__":
    main()
