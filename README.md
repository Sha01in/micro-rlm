# micro-rlm

**Recursive Language Models in ~400 lines of Python.**

A minimal, educational implementation of the RLM inference paradigm from
[Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab — MIT CSAIL, Jan 2026).

Zero external dependencies. Just Python stdlib + any OpenAI-compatible API.

Inspired by [@karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

## The Problem

LLMs suffer from **context rot**: quality degrades as prompts get longer, and there's a hard ceiling at the context window. Summarization loses detail. RAG can only fetch snippets. Sub-agent delegation is limited by autoregressive output lengths.

## The Insight

Don't feed the prompt into the neural network. **Treat it as an external object** the model interacts with programmatically.

```
┌───────────────────────────────────────────────────────┐
│                    RLM Loop                           │
│                                                       │
│  ┌──────────────┐         ┌────────────────────────┐  │
│  │   Root LLM   │◄───────►│   REPL Environment E   │  │
│  │              │  code   │                        │  │
│  │ • sees ONLY  │ ──────► │ • context (the prompt) │  │
│  │   metadata   │         │ • llm_query() function │  │
│  │ • writes code│ ◄────── │ • user variables       │  │
│  │ • reasons    │truncated│ • Python execution     │  │
│  └──────────────┘ stdout  └────────────────────────┘  │
│         │                           │                 │
│         │              ┌────────────┘                 │
│         ▼              ▼                              │
│  ┌────────────────────────────┐                       │
│  │   Sub-LLM calls            │                       │
│  │   (on programmatic slices  │                       │
│  │    of the context)         │                       │
│  └────────────────────────────┘                       │
└───────────────────────────────────────────────────────┘
```

The root LLM **never sees the full prompt**. It only gets constant-size metadata (length, a short preview). It writes code to peek into, chunk, and process slices — invoking itself recursively on each slice via `llm_query()`.

## Algorithm 1 (from the paper)

```
Input: prompt P
Output: response Y

state ← InitREPL(prompt=P)          # prompt lives in REPL, not in LLM
state ← AddFunction(state, sub_RLM) # inject recursive self-call
hist  ← [Metadata(state)]           # only metadata, not P itself

loop:
    code   ← LLM(hist)              # LLM writes code
    (state, stdout) ← REPL(code)    # execute in persistent environment
    hist   ← hist ∥ Metadata(stdout) # KEY: only truncated output goes back
    if state[Final] is set:
        return state[Final]
```

Three design choices that distinguish this from standard agents (Algorithm 2):

| | Algorithm 2 (standard) | Algorithm 1 (RLM) |
|---|---|---|
| **Prompt** | Dumped into LLM context window | External variable in REPL |
| **Output** | Generated autoregressively | Built up in REPL variables |
| **Recursion** | Verbalized sub-calls only | Programmatic loops of sub-calls |

## The Whole Algorithm in 125 Lines

Here's a complete, runnable RLM — the entire inference loop from the paper with no CLI, no logging, no extras. Just the idea. The full-featured version is [`micro_rlm.py`](micro_rlm.py).

```python
"""
The most atomic way to inference a Recursive Language Model (RLM) in pure, dependency-free Python.
This file is the complete Algorithm 1 from Zhang et al. (2026).
Everything else is just efficiency.

@karpathy-style
"""

import os       # os.environ
import re       # re.search
import io       # io.StringIO
import json     # json.dumps, json.loads
import urllib.request # urllib.request.Request, urllib.request.urlopen
import contextlib     # contextlib.redirect_stdout
import random   # random.seed, random.random, random.randint
random.seed(42) # Let there be order among chaos

# Let there be an input dataset `docs`: a massive context too large for standard LLM attention
print("generating infinite context...")
docs = []
for i in range(1000): # 1000 documents
    # 5% of the time it's an anomaly that requires semantic reasoning to detect
    if random.random() < 0.05:
        docs.append(f"Doc {i:04d} | Note: A critical system failure was narrowly avoided. | Cost: ${random.randint(100, 999)}")
    else:
        docs.append(f"Doc {i:04d} | Note: All systems operating within normal parameters. | Cost: ${random.randint(0, 99)}")
context = "\n".join(docs)
print(f"context size: {len(context)} chars")

query = "Find the exact sum of all 'Cost's in the documents that describe a critical system failure. Because the dataset is too long, chunk it and use llm_query to safely extract and sum the values."

# Let there be an LLM API, the neural engine of our system
# (Anthropic's OpenAI-compatible endpoint; works with any provider that speaks this format)
api_key = os.environ.get("ANTHROPIC_API_KEY", "sk-ant-...")
api_url = "https://api.anthropic.com/v1/chat/completions"
model_root = "claude-sonnet-4-5-20250929"  # The root reasoning model that writes code
model_sub = "claude-haiku-4-5-20251001"    # The recursive sub-call model for semantic slices

def llm(messages, model=model_root):
    """Stateless forward pass of a neural language model."""
    req = urllib.request.Request(api_url, method="POST",
        data=json.dumps({"model": model, "messages": messages, "temperature": 0.0}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]

# Let there be an Environment E: A persistent REPL holding the prompt.
# Crucially, the long context lives HERE in memory, not polluted into the LLM's context window.
class REPL:
    def __init__(self, context_data):
        def sub_call(p):
            print(f"    [sub-call] querying {model_sub} with {len(p)} chars...")
            return llm([{"role": "user", "content": p}], model_sub)
        self.namespace = {
            "context": context_data,
            # Inject recursive sub-call capability into the environment
            "llm_query": sub_call
        }
        exec("import re, json, collections", self.namespace)

    def execute(self, code):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                exec(code, self.namespace)
            except Exception as e:
                print(f"{type(e).__name__}: {e}")
        return buf.getvalue()

repl = REPL(context)

# Initialize the RLM state (Algorithm 1)
max_iters = 10
max_meta_chars = 800

# The system prompt teaches the base model how to behave recursively
sys_prompt = f"""You are a Recursive Language Model.
A vast text corpus is stored in your Python environment as `context` ({len(context)} chars).
You do NOT have direct access to it in your prompt.
You must write Python code in fenced code blocks to peek at, chunk, and process `context`.
You have a function `llm_query(prompt)` to query a sub-LLM for semantic evaluations over chunks.
IMPORTANT: You will only see the first {max_meta_chars} characters of stdout. Store large results in variables!
When you have aggregated the final answer, output: FINAL(your answer) or FINAL_VAR(variable_name)"""

hist = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": f"Query: {query}\n\nExplore `context` using python code."}
]

# Repeat in sequence (Algorithm 1 Execution Loop)
if api_key == "sk-ant-...":
    print("\nWARNING: ANTHROPIC_API_KEY environment variable not set. Please provide one.")
else:
    print("\n--- beginning rlm inference loop ---")
    for step in range(max_iters):

        # Forward the history to generate thoughts and code
        response = llm(hist)
        hist.append({"role": "assistant", "content": response})
        print(f"\n[step {step+1} LLM]\n{response.strip()}")

        # Extract ALL code blocks and execute them in the REPL
        fence = "`" * 3
        code_blocks = re.findall(fence + r"(?:python|repl)?\n(.*?)" + fence, response, re.DOTALL)
        if code_blocks:
            stdout = repl.execute("\n".join(code_blocks))

            # TRUNCATION IS KEY: this prevents context rot and forces the LLM
            # to aggregate symbolically in the REPL, not in its context window.
            stdout_trunc = stdout[:max_meta_chars] + ("\n...[truncated]" if len(stdout) > max_meta_chars else "")
            hist.append({"role": "user", "content": f"Output:\n{stdout_trunc}"})
            print(f"\n[step {step+1} REPL]\n{stdout_trunc.strip()}")
        else:
            hist.append({"role": "user", "content": "No code found. Write python or FINAL()."})

        # Check for final answer AFTER executing code (so FINAL_VAR variables exist)
        if match := re.search(r"FINAL_VAR\((.+?)\)", response):
            ans = repl.namespace.get(match.group(1), "[Variable Not Found]")
            print(f"\n[final answer (var)]\n{ans}")
            break
        if match := re.search(r"FINAL\((.*?)\)", response, re.S):
            ans = match.group(1).strip()
            print(f"\n[final answer]\n{ans}")
            break
    else:
        print("\n[RLM] exceeded maximum iterations.")
```

## Quick Start

```bash
# Set your API key (defaults to Anthropic's Claude)
export ANTHROPIC_API_KEY=sk-ant-...

# Run the census counting task (200 entries)
python micro_rlm.py --task census --n_entries 200

# Compare vanilla LLM vs RLM side by side
python micro_rlm.py --task census --n_entries 500 --compare

# Multi-hop document search
python micro_rlm.py --task search --n_entries 100

# Use Claude Opus for maximum quality
python micro_rlm.py --model claude-opus-4-6 --sub_model claude-sonnet-4-5-20250929

# Use a local model via Ollama
python micro_rlm.py --model llama3 --base_url http://localhost:11434/v1

# Use any OpenAI-compatible provider (e.g., Together, Groq, OpenAI)
python micro_rlm.py --model gpt-4o-mini --base_url https://api.openai.com/v1 --api_key sk-...

# Log trajectory to JSON for analysis or training data
python micro_rlm.py --task census --n_entries 200 --log
python micro_rlm.py --task census --n_entries 200 --log ./logs  # custom directory
```

## What You'll See

```
╔══════════════════════════════════════════════════════╗
║        micro-rlm · Recursive Language Models         ║
╚══════════════════════════════════════════════════════╝
  Task:    census (500 entries)
  Context: 38,500 chars
  Model:   claude-sonnet-4-5-20250929 (sub: claude-haiku-4-5-20251001)

── RLM (Algorithm 1: context in REPL) ───────────────

   🔄 Iteration 1/15
   💻 Code:
      lines = context.strip().split('\n')
      print(f"Total entries: {len(lines)}")
      print(lines[:3])
   📤 Output: Total entries: 500 ...

   🔄 Iteration 2/15
   💻 Code:
      chunk_size = 100
      for i in range(0, len(lines), chunk_size):
          chunk = '\n'.join(lines[i:i+chunk_size])
          counts = llm_query(f"Count people per city:\n{chunk}")
          ...
      📞 sub-call #1 (8,200 chars)
      📞 sub-call #2 (8,150 chars)
      ...

   🔄 Iteration 3/15
   ✅ FINAL (var:final_counts)

  📊 RLM Answer:
  Austin: 58
  Boston: 67
  Chicago: 61
  ...
```

## Demo Tasks

### Census Counting (linear complexity)
Like **OOLONG** from the paper — every entry must be processed. Generates N entries with random cities, asks for exact per-city counts. Vanilla LLMs fail at scale because they lose track; the RLM chunks and aggregates reliably.

### Document Search (multi-hop)
Like **BrowseComp-Plus** — clues are planted across specific documents in a sea of distractors. The RLM uses regex/keyword filtering (model priors) to narrow the search space before reading relevant documents with sub-calls.

## Trajectory Logging

Use `--log` to capture the full RLM interaction to JSON for analysis, debugging, or downstream training (the paper's RLM-Qwen3-8B was fine-tuned on 1,000 filtered trajectories — see Section 4/Appendix A).

```bash
python micro_rlm.py --task census --n_entries 200 --log           # writes rlm_census_*.json
python micro_rlm.py --task census --n_entries 200 --compare --log # also writes vanilla_census_*.json
```

Each JSON file contains: config, the query and ground truth, per-iteration deltas (root LLM response, extracted code, execution metadata, full sub-call prompts/responses), the final answer, and stats. Delta-only storage keeps file size linear rather than O(N^2). Partial trajectories are written on API errors so no data is lost.

## Key Implementation Details

**Metadata truncation** (Section 6 of the code) is the most important design choice. After each code execution, only a truncated preview of stdout goes back to the root LLM. This forces the model to store results in REPL variables instead of relying on its context window — which is what makes RLMs fundamentally different from standard coding agents.

**Sub-calls** use `llm_query()` injected into the REPL namespace. In the paper, sub-calls go to a smaller/cheaper model (e.g., GPT-5-mini for root GPT-5). By default, this uses Claude Haiku for sub-calls and Claude Sonnet for the root. Use `--sub_model` to configure this.

**FINAL detection** supports both `FINAL(direct answer text)` and `FINAL_VAR(variable_name)` which resolves a variable from the REPL. The latter is more robust for long answers.

## Security Note

The RLM executes LLM-generated Python code via `exec()` in an unsandboxed REPL. This is inherent to the design — the paper's Algorithm 1 requires a persistent code execution environment. **Do not run this on untrusted inputs or in production without sandboxing** (e.g., containers, gVisor, or a subprocess jail).

## What This Doesn't Cover

This is a **micro** implementation for learning. Production RLMs would add:

- **Sandboxed execution** — containers or subprocess isolation for the REPL
- **Async sub-calls** — the paper notes all their calls were blocking/sequential
- **Deeper recursion** — sub-calls could themselves be RLMs (depth > 1)
- **Cost controls** — budget limits, max sub-calls, timeouts
- **Native training** — the paper shows fine-tuning on RLM trajectories improves performance by 28% with just 1,000 examples

## Paper Results (for context)

| Task | Base GPT-5 | RLM(GPT-5) | Scale |
|------|-----------|-------------|-------|
| BrowseComp+ (1K docs) | 0% (exceeds window) | **91.3%** | 6-11M tokens |
| OOLONG | 44.0% | **56.5%** | 131K tokens |
| OOLONG-Pairs | 0.1% | **58.0%** | 32K tokens |
| CodeQA | 24.0% | **62.0%** | 23K-4.2M tokens |

RLMs process inputs up to **two orders of magnitude beyond the context window** while maintaining comparable or lower cost at median.

## License

MIT

## References

- [Recursive Language Models](https://arxiv.org/abs/2512.24601) — Zhang, Kraska, Khattab (MIT CSAIL, 2026)
- [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — Karpathy (2026)
- [Algorithm 1 vs Algorithm 2](https://arxiv.org/abs/2512.24601) — See Section 2 of the paper for the full comparison
