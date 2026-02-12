# CLAUDE.md

## Project Overview

**micro-rlm** is a minimal, educational, open-source implementation of Recursive Language Models (RLMs) from the paper ["Recursive Language Models"](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab — MIT CSAIL, Jan 2026). The goal is a single-file implementation that captures Algorithm 1 from the paper in ~400 lines of dependency-free Python, inspired by Karpathy's microgpt.

## Core Concept

RLMs solve long-context processing by treating the prompt as an external variable in a REPL environment rather than feeding it into the LLM's context window. The LLM writes code to peek, decompose, and recursively sub-query itself over slices of the prompt. Three design choices distinguish RLMs from standard agents:

1. **Prompt externalization** — context lives in REPL, not in LLM context window
2. **Variable-based output** — results built in REPL variables, not generated autoregressively
3. **Symbolic recursion** — sub-LLM calls made programmatically inside loops, enabling O(N) or O(N²) processing

## Architecture

Everything is in `micro_rlm.py`, organized into numbered sections:

```
Section 1: CLI & Configuration      — argparse, defaults
Section 2: LLM Interface            — urllib-based OpenAI-compatible API calls (no deps)
Section 3: REPL Environment          — persistent Python REPL with `context` var and `llm_query()` fn
Section 4: Parsing                   — extract ```repl code blocks and FINAL/FINAL_VAR markers
Section 5: System Prompt             — teaches the LLM to operate as an RLM
Section 6: RLM Loop                  — Algorithm 1 implementation (the core)
Section 7: Vanilla Baseline          — direct LLM call for comparison (Algorithm 2)
Section 8: Demo Tasks                — census counting (linear) and document search (multi-hop)
Section 9: Main                      — CLI entry point, runs demo and optional comparison
```

## Key Design Decisions

- **Zero dependencies** — only Python stdlib + urllib for API calls. This is intentional; don't add requests, openai, etc.
- **Single file** — everything in `micro_rlm.py`. Keep it this way for the microgpt ethos.
- **Metadata truncation** — after REPL execution, only truncated stdout goes back to the root LLM (controlled by `--metadata_chars`). This is THE critical design choice that makes it an RLM vs a standard coding agent. See the comment block in Section 6.
- **Stats tracking** — the `stats` dict tracks root_calls, sub_calls, input/output chars for cost analysis.
- **Deterministic tasks** — demo tasks use seeded RNG so ground truth is reproducible.

## Running

```bash
# Basic run
export ANTHROPIC_API_KEY=sk-ant-...
python micro_rlm.py --task census --n_entries 300

# Compare vanilla vs RLM
python micro_rlm.py --task census --n_entries 500 --compare

# Different model/provider
python micro_rlm.py --model llama3 --base_url http://localhost:11434/v1
```

## Development Guidelines

### What to preserve
- Zero external dependencies (stdlib + urllib only)
- Single-file architecture
- Clear section numbering and comments tying back to the paper
- The Algorithm 1 comment block in Section 6
- Deterministic seeded task generation with verifiable ground truth

### Code style
- Functions over classes (except REPL which needs state)
- Descriptive variable names; avoid abbreviations except established ones (LLM, REPL, RLM)
- Comments should reference the paper where relevant (e.g., "Algorithm 1, line 2")
- Use f-strings, not .format() or %

### Testing
- `python3 -c "import ast; ast.parse(open('micro_rlm.py').read())"` — syntax check
- `python3 micro_rlm.py` without API key should print usage and exit cleanly
- Task generators can be tested without API access by importing and calling directly
- Ground truth for census task at seed=42, n=500: Austin:63, Boston:81, Chicago:60, Denver:61, Miami:56, Phoenix:61, Portland:57, Seattle:61

### Common modifications expected
- **New demo tasks** — add to Section 8. Each task function returns `(context, query, truth)`. Task complexity should be categorizable as O(1), O(N), or O(N²) per the paper's framework.
- **Async sub-calls** — the paper notes this as a key optimization. Would use asyncio from stdlib.
- **Cost tracking** — currently tracks chars, could estimate tokens (~4 chars/token).
- **Trajectory logging** — recording full RLM trajectories to JSON for analysis or training data generation (per paper's Section 4/Appendix A).

## Paper Reference

Key sections of the paper mapped to this implementation:

| Paper Section | Implementation |
|---|---|
| Algorithm 1 (§2) | `rlm()` function in Section 6 |
| Algorithm 2 (§2) | `vanilla_llm()` in Section 7 |
| REPL environment E | `REPL` class in Section 3 |
| Metadata truncation | `truncate()` + the truncation in the rlm loop |
| OOLONG (linear task) | `make_census_task()` in Section 8 |
| BrowseComp-Plus (search) | `make_search_task()` in Section 8 |
| System prompt (Appendix C) | `SYSTEM_PROMPT` in Section 5 |
| FINAL/FINAL_VAR detection | `extract_final()` in Section 4 |
