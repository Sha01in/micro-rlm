"""
micro_rlm.py — Recursive Language Models in ~400 lines of Python.

A minimal, educational implementation of the RLM inference paradigm from:
"Recursive Language Models" (Zhang, Kraska, Khattab, 2026)
Paper: https://arxiv.org/abs/2512.24601

The key insight: don't feed long prompts into the LLM directly. Instead,
treat the prompt as an external variable in a REPL environment and let
the LLM write code to peek, decompose, and recursively process slices.

This file captures Algorithm 1 from the paper. Zero external dependencies —
only Python stdlib + any OpenAI-compatible API endpoint (defaults to Anthropic's Claude).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python micro_rlm.py --task census --n_entries 500
    python micro_rlm.py --task census --n_entries 500 --compare

Inspired by @karpathy's microgpt.
"""

import os, re, sys, json, time, random, argparse, traceback, urllib.request, urllib.error
from contextlib import redirect_stdout
from io import StringIO
from collections import Counter

# ════════════════════════════════════════════════════════════════
# Section 1: CLI & Configuration
# ════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="micro-rlm: Recursive Language Models")
parser.add_argument("--model", default="claude-sonnet-4-5-20250929", help="Root LLM model")
parser.add_argument("--sub_model", default=None, help="Sub-call model (default: claude-haiku-4-5-20251001)")
parser.add_argument("--base_url", default="https://api.anthropic.com/v1/", help="API base URL")
parser.add_argument("--api_key", default=None, help="API key (or set ANTHROPIC_API_KEY env var)")
parser.add_argument("--max_iters", type=int, default=15, help="Max RLM loop iterations")
parser.add_argument("--metadata_chars", type=int, default=800, help="Max stdout chars shown to root")
parser.add_argument("--task", default="census", choices=["census", "search"], help="Demo task")
parser.add_argument("--n_entries", type=int, default=200, help="Entries in demo task")
parser.add_argument("--compare", action="store_true", help="Also run vanilla LLM for comparison")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--quiet", action="store_true", help="Less verbose output")
if __name__ == "__main__":
    args = parser.parse_args()
else:
    # Sensible defaults when imported as a library
    args = argparse.Namespace(
        model="claude-sonnet-4-5-20250929", sub_model=None,
        base_url="https://api.anthropic.com/v1/",
        api_key=None, max_iters=15, metadata_chars=800, task="census",
        n_entries=200, compare=False, seed=42, quiet=False,
    )

API_KEY = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
SUB_MODEL = args.sub_model or "claude-haiku-4-5-20251001"

# ════════════════════════════════════════════════════════════════
# Section 2: LLM Interface (zero deps — uses urllib)
# ════════════════════════════════════════════════════════════════

stats = {"root_calls": 0, "sub_calls": 0, "input_chars": 0, "output_chars": 0}

def llm_call(messages, model=None):
    """Call an OpenAI-compatible chat completions endpoint."""
    model = model or args.model
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.0}).encode()
    req = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        hint = " — check your API key" if e.code in (401, 403) else ""
        print(f"API error {e.code}{hint}: {e.read().decode()[:200]}", file=sys.stderr)
        raise
    except urllib.error.URLError as e:
        print(f"Connection error ({args.base_url}): {e.reason}", file=sys.stderr)
        raise
    text = result["choices"][0]["message"]["content"]
    stats["input_chars"] += sum(len(m.get("content", "")) for m in messages)
    stats["output_chars"] += len(text)
    return text

# ════════════════════════════════════════════════════════════════
# Section 3: REPL Environment — the "E" in Algorithm 1
# ════════════════════════════════════════════════════════════════

class REPL:
    """Persistent Python REPL that holds the prompt as a variable.

    This is the core of the RLM idea: the prompt lives HERE as a string
    variable, not in the LLM's context window. The LLM interacts with it
    only through code execution.
    """

    def __init__(self, context, llm_query_fn=None):
        self.namespace = {"context": context}
        if llm_query_fn:
            self.namespace["llm_query"] = llm_query_fn
        # Pre-import useful modules so the LLM can use them immediately
        exec("import re, math, json\nfrom collections import Counter, defaultdict", self.namespace)

    def execute(self, code):
        """Execute code, capturing stdout. Returns (stdout_str, had_error)."""
        buf = StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, self.namespace)
            return buf.getvalue(), False
        except Exception:
            traceback.print_exc(file=buf)
            return buf.getvalue(), True

    def user_vars(self):
        """Return dict of user-defined variable names -> type names."""
        skip = {"__builtins__", "re", "math", "json", "Counter",
                "defaultdict", "context", "llm_query"}
        return {k: type(v).__name__ for k, v in self.namespace.items()
                if k not in skip and not k.startswith("_")}

# ════════════════════════════════════════════════════════════════
# Section 4: Parsing — extract code blocks and FINAL markers
# ════════════════════════════════════════════════════════════════

def extract_code(text):
    """Extract ```repl or ```python code blocks from LLM output."""
    blocks = re.findall(r"```(?:repl|python)?\s*\n(.*?)```", text, re.DOTALL)
    return "\n".join(blocks) if blocks else None

def extract_final(text):
    """Detect FINAL(answer) or FINAL_VAR(var_name) in LLM output.
    Returns ('text', answer_str) or ('var', var_name) or None.
    Note: FINAL() matches single-line answers only; use FINAL_VAR() for multi-line."""
    m = re.search(r"FINAL_VAR\((\w+)\)", text)
    if m:
        return "var", m.group(1)
    m = re.search(r"FINAL\((.+)\)\s*$", text, re.MULTILINE)
    if m:
        return "text", m.group(1).strip()
    return None

def truncate(text, max_chars):
    """Truncate to max_chars, keeping start and end for context."""
    if len(text) <= max_chars:
        return text
    h = max_chars // 2
    return f"{text[:h]}\n... [{len(text)} chars total, truncated] ...\n{text[-h:]}"

# ════════════════════════════════════════════════════════════════
# Section 5: System Prompt — teaches the LLM to be an RLM
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an RLM (Recursive Language Model). You must answer a query about
a large context stored in a REPL environment — NOT in your context window.

## Your REPL has:
- `context` — string variable with the input data ({n_chars} chars total)
- `llm_query(prompt)` — call a sub-LLM for semantic reasoning (~100K char limit)
- `print()` — view results (WARNING: output is truncated to ~{meta_chars} chars)

## How to work:
1. Peek at the context: `print(context[:500])`
2. Understand its structure, then design a chunking strategy
3. For semantic tasks, use `llm_query()` on chunks. Batch generously.
4. Store intermediate results in variables — don't rely on print output
5. Aggregate results and call FINAL() or FINAL_VAR()

## Output format:
- Write code inside ```repl blocks
- When done: FINAL(your answer here) or FINAL_VAR(variable_name)
- Do NOT use FINAL until your analysis is complete

## Example — counting items in a long list:
```repl
lines = context.strip().split('\\n')
print(f"Total lines: {{len(lines)}}")
print(lines[:3])  # peek at structure
```
Then in the next step:
```repl
chunk_size = 200
results = []
for i in range(0, len(lines), chunk_size):
    chunk = '\\n'.join(lines[i:i+chunk_size])
    ans = llm_query(f"Count items matching X in this data:\\n{{chunk}}")
    results.append(ans)
    print(f"Chunk {{i//chunk_size}}: {{ans}}")
```
Then aggregate and finalize.
"""

# ════════════════════════════════════════════════════════════════
# Section 6: The RLM Loop — Algorithm 1 from the paper
# ════════════════════════════════════════════════════════════════
#
#   Input: prompt P
#   state ← InitREPL(prompt=P)
#   state ← AddFunction(state, sub_RLM)
#   hist  ← [Metadata(state)]
#   loop:
#       code          ← LLM(hist)
#       (state, stdout) ← REPL(state, code)
#       hist          ← hist ∥ code ∥ Metadata(stdout)   # <-- truncated!
#       if state[Final] is set: return state[Final]
#
# ════════════════════════════════════════════════════════════════

def rlm(context, query):
    """The Recursive Language Model inference loop.

    Context is placed in a REPL variable — the root LLM never sees it
    directly. It writes code to peek, chunk, and sub-query over slices.
    """

    # 1. Create the sub-call function injected into the REPL
    def llm_query(prompt):
        stats["sub_calls"] += 1
        if not args.quiet:
            print(f"      📞 sub-call #{stats['sub_calls']} ({len(prompt):,} chars)")
        return llm_call([{"role": "user", "content": prompt}], model=SUB_MODEL)

    # 2. Initialize REPL with context as variable (Algorithm 1, line 1-2)
    repl = REPL(context, llm_query_fn=llm_query)

    # 3. Build initial metadata — constant size, NOT the full context
    system = SYSTEM_PROMPT.format(n_chars=len(context), meta_chars=args.metadata_chars)
    preview = context[:400] if isinstance(context, str) else str(context)[:400]
    history = [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"Query: {query}\n\n"
            f"Context loaded in REPL as `context` ({len(context):,} chars).\n"
            f"First 400 chars: {preview}...\n\n"
            f"Write ```repl code to explore and answer the query."
        )},
    ]

    # 4. RLM loop (Algorithm 1, lines 4-8)
    for it in range(1, args.max_iters + 1):
        if not args.quiet:
            print(f"\n   🔄 Iteration {it}/{args.max_iters}")

        # Ask root LLM for next action
        stats["root_calls"] += 1
        response = llm_call(history, model=args.model)
        history.append({"role": "assistant", "content": response})

        # Execute any code blocks
        code = extract_code(response)
        if code:
            if not args.quiet:
                lines = code.split("\n")
                preview = "\n".join(lines[:8])
                if len(lines) > 8:
                    preview += f"\n      ... ({len(lines)} lines total)"
                print(f"   💻 Code:\n      {preview.replace(chr(10), chr(10) + '      ')}")

            stdout, had_error = repl.execute(code)

            # ── KEY DESIGN CHOICE ──────────────────────────────────
            # Only TRUNCATED metadata goes back to the LLM. This is
            # what forces the model to store results in REPL variables
            # instead of polluting its context window with raw output.
            # Without this, the approach degenerates to Algorithm 2.
            # ───────────────────────────────────────────────────────
            meta = f"[Execution {'ERROR' if had_error else 'OK'}]\n{truncate(stdout, args.metadata_chars)}"
            uvars = repl.user_vars()
            if uvars:
                meta += f"\n[REPL variables: {uvars}]"
            history.append({"role": "user", "content": meta})

            if not args.quiet and stdout.strip():
                print(f"   📤 Output: {truncate(stdout.strip(), 200)}")

        # Check for FINAL answer (after executing code so variables are set)
        final = extract_final(response)
        if final:
            kind, value = final
            if kind == "var":
                answer = str(repl.namespace.get(value, f"[Variable '{value}' not found]"))
            else:
                answer = value
            if not args.quiet:
                label = f"var:{value}" if kind == "var" else "direct"
                print(f"   ✅ FINAL ({label})")
            return answer

        # If no code and no FINAL, nudge the model
        if not code:
            history.append({"role": "user", "content":
                "No code detected. Write ```repl code to continue, or FINAL(answer)."})

    return "[Exceeded max iterations without a final answer]"

# ════════════════════════════════════════════════════════════════
# Section 7: Vanilla Baseline (for comparison)
# ════════════════════════════════════════════════════════════════

def vanilla_llm(context, query):
    """Baseline: stuff context + query directly into the LLM. (Algorithm 2, Flaw #1)"""
    stats["root_calls"] += 1
    return llm_call([{"role": "user", "content":
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer concisely:"}])

# ════════════════════════════════════════════════════════════════
# Section 8: Demo Tasks with Ground Truth
# ════════════════════════════════════════════════════════════════

def make_census_task(n_entries=200, n_cities=8, seed=42):
    """Census counting — every entry must be processed (linear complexity).

    This is analogous to OOLONG from the paper: the answer depends on
    semantically transforming and aggregating every line of input.
    """
    rng = random.Random(seed)
    cities = ["Portland", "Seattle", "Denver", "Austin",
              "Chicago", "Boston", "Miami", "Phoenix"][:n_cities]
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
             "Grace", "Henry", "Iris", "Jack", "Karen", "Leo",
             "Maya", "Noah", "Olivia", "Paul", "Quinn", "Rose"]
    occupations = ["Engineer", "Teacher", "Doctor", "Artist",
                   "Chef", "Writer", "Nurse", "Pilot"]

    entries, city_counts = [], Counter()
    for i in range(n_entries):
        city = rng.choice(cities)
        entries.append(
            f"ID: {i+1:04d} | Name: {rng.choice(names)} | "
            f"Age: {rng.randint(20, 65)} | City: {city} | "
            f"Occupation: {rng.choice(occupations)}"
        )
        city_counts[city] += 1

    context = "\n".join(entries)
    query = ("Count the exact number of people in each city. "
             "Return ONLY 'City: count' pairs, one per line, sorted alphabetically.")
    truth = "\n".join(f"{c}: {n}" for c, n in sorted(city_counts.items()))
    return context, query, truth


def make_search_task(n_docs=100, seed=42):
    """Multi-hop search — find and connect clues across documents.

    Analogous to BrowseComp-Plus: most documents are distractors,
    the answer requires connecting info from specific documents.
    """
    rng = random.Random(seed)
    topics = ["quantum computing", "marine biology", "urban planning",
              "medieval history", "renewable energy", "jazz music"]
    docs = []
    for i in range(n_docs):
        filler = " ".join(rng.choices(
            ["research", "shows", "that", "the", "field", "of", topics[i % len(topics)],
             "has", "advanced", "significantly", "in", "recent", "years", "with",
             "new", "methods", "and", "approaches", "being", "developed", "across",
             "multiple", "institutions", "worldwide"], k=rng.randint(40, 80)))
        docs.append(f"[Document {i+1}] Topic: {rng.choice(topics)}\n{filler}\n")

    # Plant clues
    d1, d2 = n_docs // 3, 2 * n_docs // 3
    docs[d1] = (
        f"[Document {d1+1}] Topic: renewable energy\n"
        f"The Helios Project, launched in 2019 by Dr. Sarah Chen at MIT, achieved "
        f"a breakthrough in perovskite solar cell efficiency, reaching 33.7% "
        f"conversion rate. The internal code name was 'Phoenix Rising'.\n"
    )
    docs[d2] = (
        f"[Document {d2+1}] Topic: renewable energy\n"
        f"Following up on the Phoenix Rising project (ref: Document {d1+1}), "
        f"Dr. Chen's team received the 2023 Nobel Prize in Chemistry for their "
        f"33.7% perovskite efficiency milestone.\n"
    )

    context = "\n".join(docs)
    query = ("What was the code name of Dr. Sarah Chen's solar cell project, "
             "and what prize did her team receive?")
    truth = "Code name: Phoenix Rising. Prize: 2023 Nobel Prize in Chemistry."
    return context, query, truth

# ════════════════════════════════════════════════════════════════
# Section 9: Main — run demo and optionally compare
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not API_KEY:
        print("Set ANTHROPIC_API_KEY or pass --api_key. Works with any OpenAI-compatible API.")
        print("Examples:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  python micro_rlm.py --task census --n_entries 300")
        print("  python micro_rlm.py --task census --n_entries 300 --compare")
        print("  python micro_rlm.py --model ollama/llama3 --base_url http://localhost:11434/v1")
        sys.exit(1)

    # Build task
    if args.task == "census":
        context, query, truth = make_census_task(args.n_entries, seed=args.seed)
    else:
        context, query, truth = make_search_task(args.n_entries, seed=args.seed)

    print("╔══════════════════════════════════════════════════════╗")
    print("║        micro-rlm · Recursive Language Models        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Task:    {args.task} ({args.n_entries} entries)")
    print(f"  Context: {len(context):,} chars")
    print(f"  Model:   {args.model} (sub: {SUB_MODEL})")
    print(f"  Query:   {query[:70]}...")
    print(f"  Truth:   {truth[:70]}...")

    # ── Vanilla baseline ──
    if args.compare:
        print("\n── Vanilla LLM (Algorithm 2: context in window) ─────")
        for k in stats:
            stats[k] = 0
        t0 = time.time()
        try:
            v_answer = vanilla_llm(context, query)
        except Exception as e:
            v_answer = f"[Error: {e}]"
        print(f"  Answer: {v_answer[:300]}")
        print(f"  Time: {time.time()-t0:.1f}s | Calls: {stats['root_calls']}")
        print(f"  Input chars sent to LLM: {stats['input_chars']:,}")

    # ── RLM ──
    print("\n── RLM (Algorithm 1: context in REPL) ───────────────")
    for k in stats:
        stats[k] = 0
    t0 = time.time()
    answer = rlm(context, query)
    elapsed = time.time() - t0

    print(f"\n  {'─'*50}")
    print(f"  📊 RLM Answer:\n  {answer[:500]}")
    print(f"\n  🎯 Ground Truth:\n  {truth[:500]}")
    print(f"\n  ⏱  Time: {elapsed:.1f}s")
    print(f"  📞 Root calls: {stats['root_calls']} | Sub-calls: {stats['sub_calls']}")
    print(f"  📨 Chars in: {stats['input_chars']:,} | out: {stats['output_chars']:,}")
