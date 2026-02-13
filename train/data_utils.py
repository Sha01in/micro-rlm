"""Shared data utilities for the micro-rlm training pipeline.

Handles trajectory loading, filtering, ChatML conversion, sub-call
extraction, reward labeling, and tokenization with loss masking.

No JAX dependency — pure Python + transformers tokenizer.
"""

import os
import sys
import json
import pathlib

# Add parent dir so we can import micro_rlm task generators
_PARENT = str(pathlib.Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from micro_rlm import make_census_task, make_search_task, SYSTEM_PROMPT


# ── Trajectory loading ──────────────────────────────────────────

def load_trajectories(data_dir, mode="rlm"):
    """Load all trajectory JSONs from a directory.

    Args:
        data_dir: Directory containing trajectory JSON files from --log.
        mode: Filter by mode ("rlm", "vanilla", or None for all).
    Returns:
        List of trajectory dicts sorted by timestamp.
    """
    trajs = []
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            try:
                traj = json.load(f)
            except json.JSONDecodeError:
                print(f"  Warning: skipping malformed JSON: {fname}")
                continue
        if mode and traj.get("mode") != mode:
            continue
        traj["_path"] = path
        trajs.append(traj)
    trajs.sort(key=lambda t: t.get("timestamp", ""))
    return trajs


# ── Filtering ───────────────────────────────────────────────────

def _normalize(text):
    """Normalize answer text for comparison (lowercase, collapse whitespace)."""
    return " ".join(str(text).lower().split())


def filter_correct(trajs):
    """Keep only trajectories where answer matches ground_truth."""
    return [
        t for t in trajs
        if t.get("answer") and t.get("ground_truth")
        and _normalize(t["answer"]) == _normalize(t["ground_truth"])
    ]


# ── Context regeneration from task seed ─────────────────────────

_TASK_GENERATORS = {
    "census": make_census_task,
    "search": make_search_task,
}


def regenerate_context(traj):
    """Regenerate full context string from task config seed.

    Works for built-in tasks (census, search) by re-running the
    deterministic task generator with the same seed/params.
    Returns None for custom tasks.
    """
    task = traj.get("task")
    config = traj.get("config", {})
    gen = _TASK_GENERATORS.get(task)
    if not gen:
        return None
    kwargs = {"seed": config.get("seed", 42)}
    if task == "census":
        kwargs["n_entries"] = config.get("n_entries", 200)
    elif task == "search":
        kwargs["n_docs"] = config.get("n_entries", 100)
    context, _, _ = gen(**kwargs)
    return context


# ── ChatML conversion for root model SFT ───────────────────────

def traj_to_root_messages(traj):
    """Convert a trajectory to multi-turn ChatML messages for root model SFT.

    Reconstructs the exact conversation that produced the trajectory:
      system prompt → user query → (assistant code → user metadata)* → assistant FINAL

    Returns list of {role, content} dicts, or None if malformed.
    """
    config = traj.get("config", {})
    context = regenerate_context(traj)
    if context is None:
        return None

    n_chars = traj.get("context_chars", len(context))
    meta_chars = config.get("metadata_chars", 800)

    # System message
    system = SYSTEM_PROMPT.format(n_chars=n_chars, meta_chars=meta_chars)

    # Initial user message with context preview (matches rlm() exactly)
    preview = context[:400]
    query = traj.get("query", "")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"Query: {query}\n\n"
            f"Context loaded in REPL as `context` ({n_chars:,} chars).\n"
            f"First 400 chars: {preview}...\n\n"
            f"Write ```repl code to explore and answer the query."
        )},
    ]

    # Add each iteration (assistant response + execution metadata)
    for it in traj.get("iterations", []):
        root_resp = it.get("root_response")
        if root_resp:
            messages.append({"role": "assistant", "content": root_resp})

        exec_info = it.get("execution")
        if exec_info and exec_info.get("metadata_sent"):
            messages.append({"role": "user", "content": exec_info["metadata_sent"]})

    # Conversation must end with assistant turn for SFT
    if not messages or messages[-1]["role"] != "assistant":
        return None

    return messages


# ── Sub-call extraction for sub model SFT ───────────────────────

def extract_sub_calls(trajs):
    """Flatten sub-calls across all trajectories into single-turn chat pairs.

    Each sub-call becomes a standalone (user → assistant) exchange.
    Returns list of {"messages": [{role, content}, {role, content}]}.
    """
    pairs = []
    for traj in trajs:
        for it in traj.get("iterations", []):
            for sc in it.get("sub_calls", []):
                prompt = sc.get("prompt", "")
                response = sc.get("response", "")
                if prompt and response:
                    pairs.append({
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response},
                        ]
                    })
    return pairs


# ── Reward labeling ─────────────────────────────────────────────

def label_trajectories(trajs):
    """Label trajectories for reward model training.

    For each trajectory, creates per-iteration samples:
      - conversation up to iteration N → label
      - label = outcome (1.0 correct, 0.0 wrong) × step penalty

    Step penalties:
      - Error step: 0.3×
      - Empty output step: 0.1×
      - Normal step: 1.0×

    Returns list of {"messages": [...], "label": float}.
    """
    samples = []
    for traj in trajs:
        messages = traj_to_root_messages(traj)
        if not messages:
            continue

        correct = (
            traj.get("answer") and traj.get("ground_truth")
            and _normalize(traj["answer"]) == _normalize(traj["ground_truth"])
        )
        outcome = 1.0 if correct else 0.0

        # Build prefix messages incrementally
        prefix = list(messages[:2])  # system + initial user

        for it in traj.get("iterations", []):
            root_resp = it.get("root_response")
            if root_resp:
                prefix.append({"role": "assistant", "content": root_resp})

            # Compute step penalty
            exec_info = it.get("execution")
            penalty = 1.0
            if exec_info:
                if exec_info.get("had_error"):
                    penalty = 0.3
                elif not exec_info.get("stdout", "").strip():
                    penalty = 0.1
                if exec_info.get("metadata_sent"):
                    prefix.append({"role": "user", "content": exec_info["metadata_sent"]})

            label = outcome * penalty
            samples.append({
                "messages": [dict(m) for m in prefix],
                "label": label,
            })

    return samples


# ── Tokenization with loss masking ──────────────────────────────

def _subseq_match(seq, start, pattern):
    """Check if pattern matches seq starting at position start."""
    if start + len(pattern) > len(seq):
        return False
    return seq[start:start + len(pattern)] == pattern


def tokenize_chat(messages, tokenizer, max_len, mask_user=True):
    """Tokenize ChatML messages with loss masking on assistant tokens only.

    Uses the tokenizer's chat template (e.g., Qwen/ChatML format)
    to produce properly formatted token sequences.

    Args:
        messages: List of {role, content} dicts.
        tokenizer: HuggingFace tokenizer with apply_chat_template.
        max_len: Maximum sequence length (tokens).
        mask_user: If True, set labels to -100 for non-assistant tokens.

    Returns:
        {"input_ids": list[int], "labels": list[int]} or None.
        Masked positions have label = -100.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    tokens = tokenizer(text, truncation=True, max_length=max_len)
    input_ids = tokens["input_ids"]

    if len(input_ids) < 4:
        return None

    if not mask_user:
        return {"input_ids": input_ids, "labels": list(input_ids)}

    # Build labels: -100 for non-assistant tokens
    labels = [-100] * len(input_ids)

    # Find assistant regions using ChatML markers:
    #   <|im_start|>assistant\n ... <|im_end|>
    # This is Qwen/ChatML specific — adjust for other templates.
    assistant_start = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    assistant_end = tokenizer.encode(
        "<|im_end|>", add_special_tokens=False
    )

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

    # Verify we found at least some assistant tokens
    if all(l == -100 for l in labels):
        return None

    return {"input_ids": input_ids, "labels": labels}


# ── Dataset batching ────────────────────────────────────────────

def pad_and_batch(samples, batch_size, pad_id=0):
    """Pad samples to uniform length within each batch.

    Args:
        samples: List of {"input_ids": [...], "labels": [...]}.
        batch_size: Samples per batch.
        pad_id: Padding token ID.

    Returns:
        List of batch dicts with keys:
          input_ids: [[...]], labels: [[...]], attention_mask: [[...]]
    """
    batches = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        max_len = max(len(s["input_ids"]) for s in batch)

        input_ids, labels, masks = [], [], []
        for s in batch:
            pad_len = max_len - len(s["input_ids"])
            input_ids.append(s["input_ids"] + [pad_id] * pad_len)
            labels.append(s["labels"] + [-100] * pad_len)
            masks.append([1] * len(s["input_ids"]) + [0] * pad_len)

        batches.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": masks,
        })

    return batches


# ── Summary / stats ─────────────────────────────────────────────

def print_data_summary(trajs, label="trajectories"):
    """Print summary statistics for a trajectory dataset."""
    n_iters = sum(len(t.get("iterations", [])) for t in trajs)
    n_sub = sum(
        len(sc)
        for t in trajs
        for it in t.get("iterations", [])
        for sc in [it.get("sub_calls", [])]
    )
    print(f"  {label}: {len(trajs)} trajectories, {n_iters} iterations, {n_sub} sub-calls")
