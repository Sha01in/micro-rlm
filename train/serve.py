"""Multi-model OpenAI-compatible server with JAX JIT inference.

Serves fine-tuned root, sub-call, and reward models via a single HTTP
endpoint compatible with micro_rlm.py's --base_url flag.

Usage:
    # Serve root model only
    python train/serve.py --root-model ./models/root_v1 --port 8000

    # Serve root + sub model
    python train/serve.py --root-model ./models/root_v1 \\
        --sub-model ./models/sub_v1 --port 8000

    # Then use with micro_rlm.py
    python micro_rlm.py --model root --sub_model sub \\
        --base_url http://localhost:8000/v1 --api_key dummy

Endpoint: POST /v1/chat/completions (OpenAI-compatible)
Routes requests to the correct model based on the "model" field.
"""

import argparse
import json
import os
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import orbax.checkpoint as ocp
from flax import nnx
from transformers import AutoTokenizer
from easydel import AutoEasyDeLModelForCausalLM

from sft_root import LoRAParam, inject_lora, make_mesh


# ── CLI ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Serve fine-tuned micro-rlm models")
    p.add_argument("--root-model", default=None,
                   help="Path to root model directory (from sft_root.py)")
    p.add_argument("--sub-model", default=None,
                   help="Path to sub-call model directory (from sft_sub.py)")
    p.add_argument("--reward-model", default=None,
                   help="Path to reward model directory (from reward_model.py)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


# ── Model loading ───────────────────────────────────────────────

BUCKETS = [256, 512, 1024, 2048, 4096]


def find_bucket(length):
    """Find the smallest bucket that fits the given length."""
    for b in BUCKETS:
        if b >= length:
            return b
    return length


class LoadedModel:
    """A loaded model ready for inference.

    Loads the base model, injects LoRA, restores LoRA weights,
    and provides a generate() method for autoregressive decoding.
    """

    def __init__(self, model_dir, model_type="root"):
        self.model_dir = model_dir
        self.model_type = model_type

        config_path = os.path.join(model_dir, "train_config.json")
        with open(config_path) as f:
            self.config = json.load(f)

        base_model = self.config["base_model"]
        print(f"  Loading {model_type} model: {base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model (EasyDel 0.1.x returns NNX module)
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            base_model,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            auto_shard_model=False,
        )

        # Disable precomputed causal mask to save ~1 GB GPU memory
        model.config.precompute_masks = False

        # Inject LoRA structure
        lora_rank = self.config.get("lora_rank", 16)
        lora_alpha = self.config.get("lora_alpha", 32.0)
        inject_lora(model, lora_rank, lora_alpha, jax.random.PRNGKey(0))

        # Split, restore LoRA weights, merge back for inference
        graphdef, lora_state, base_state = nnx.split(model, LoRAParam, ...)

        lora_path = os.path.join(model_dir, "lora_params")
        if os.path.exists(lora_path):
            checkpointer = ocp.PyTreeCheckpointer()
            lora_state = checkpointer.restore(lora_path, item=lora_state)
            print(f"    Restored LoRA weights (rank={lora_rank}, alpha={lora_alpha})")

        # Merge back into a single model for fast inference
        self.model = nnx.merge(graphdef, lora_state, base_state)
        self.mesh = make_mesh()

        # Warm up JIT with a dummy forward pass
        print(f"    Warming up JIT...")
        dummy = jnp.ones((1, 16), dtype=jnp.int32)
        with self.mesh:
            self.model(input_ids=dummy, attention_mask=dummy)
        print(f"    {model_type} model ready")

    def generate(self, messages, max_tokens=512, temperature=0.0):
        """Generate a response given chat messages.

        Uses the tokenizer's chat template, then autoregressive
        decoding with bucket-based static shapes for JIT stability.
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer.encode(text)
        prompt_len = len(tokens)
        total_len = prompt_len + max_tokens
        bucket = find_bucket(total_len)

        # Pad to bucket for stable JIT shapes
        pad_len = bucket - prompt_len
        input_ids = jnp.array(
            [tokens + [self.tokenizer.pad_token_id] * pad_len], dtype=jnp.int32
        )
        attention_mask = jnp.array(
            [[1] * prompt_len + [0] * pad_len], dtype=jnp.int32
        )

        # Autoregressive generation
        generated = []
        eos_id = self.tokenizer.eos_token_id

        for i in range(max_tokens):
            pos = prompt_len + i
            if pos >= bucket:
                break

            with self.mesh:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, pos - 1]

            if temperature > 0:
                probs = jax.nn.softmax(logits / temperature)
                next_id = int(jax.random.categorical(
                    jax.random.PRNGKey(int(time.time() * 1000) % 2**31),
                    jnp.log(probs),
                ))
            else:
                next_id = int(jnp.argmax(logits))

            if next_id == eos_id:
                break

            generated.append(next_id)
            input_ids = input_ids.at[0, pos].set(next_id)
            attention_mask = attention_mask.at[0, pos].set(1)

        return self.tokenizer.decode(generated, skip_special_tokens=True)


# ── Model registry ──────────────────────────────────────────────

class ModelRegistry:
    """Holds all loaded models and routes requests by model name."""

    def __init__(self):
        self.models = {}
        self.aliases = {}
        self.default_model = None

    def register(self, name, model, aliases=None):
        self.models[name] = model
        if aliases:
            for alias in aliases:
                self.aliases[alias] = name
        if self.default_model is None:
            self.default_model = name

    def get(self, model_name):
        if model_name in self.models:
            return self.models[model_name]
        canonical = self.aliases.get(model_name)
        if canonical:
            return self.models[canonical]
        return None

    def list_models(self):
        return [
            {"id": name, "object": "model", "owned_by": "micro-rlm"}
            for name in self.models
        ]


# ── HTTP handler ────────────────────────────────────────────────

registry = ModelRegistry()


class ChatHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible /v1/chat/completions endpoint."""

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat()
        else:
            self._error(404, f"Not found: {self.path}")

    def do_GET(self):
        if self.path == "/v1/models":
            self._handle_models()
        elif self.path == "/health":
            self._json_response(200, {"status": "ok"})
        else:
            self._error(404, f"Not found: {self.path}")

    def _handle_chat(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        except (json.JSONDecodeError, ValueError) as e:
            self._error(400, f"Invalid JSON: {e}")
            return

        model_name = body.get("model", registry.default_model)
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.0)

        model = registry.get(model_name)
        if model is None:
            available = list(registry.models.keys()) + list(registry.aliases.keys())
            self._error(404, f"Model '{model_name}' not found. Available: {available}")
            return

        t0 = time.time()
        try:
            content = model.generate(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            self._error(500, f"Generation error: {e}")
            return
        elapsed = time.time() - t0

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        self._json_response(200, response)
        print(f"  [{model_name}] {elapsed:.1f}s — {len(content)} chars")

    def _handle_models(self):
        self._json_response(200, {"object": "list", "data": registry.list_models()})

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, message):
        self._json_response(code, {
            "error": {"message": message, "type": "invalid_request_error", "code": code}
        })

    def log_message(self, format, *a):
        pass


# ── Main ────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not any([args.root_model, args.sub_model, args.reward_model]):
        print("No models specified. Pass at least one of:")
        print("  --root-model ./models/root_v1")
        print("  --sub-model  ./models/sub_v1")
        print("  --reward-model ./models/reward_v1")
        sys.exit(1)

    print("Loading models...")

    if args.root_model:
        root = LoadedModel(args.root_model, "root")
        registry.register("root", root, aliases=[
            "qwen-3b-rlm", "root-rlm",
            root.config.get("base_model", ""),
        ])

    if args.sub_model:
        sub = LoadedModel(args.sub_model, "sub")
        registry.register("sub", sub, aliases=[
            "qwen-1.5b-sub", "sub-rlm",
            sub.config.get("base_model", ""),
        ])

    if args.reward_model:
        reward = LoadedModel(args.reward_model, "reward")
        registry.register("reward", reward, aliases=["reward-rlm"])

    print(f"\nModels loaded: {list(registry.models.keys())}")
    print(f"Aliases: {dict(registry.aliases)}")
    print(f"\nServer: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"\nUsage:")

    model_names = list(registry.models.keys())
    root_name = model_names[0] if model_names else "root"
    print(f"  python micro_rlm.py --model {root_name} "
          f"--base_url http://localhost:{args.port}/v1 --api_key dummy")

    server = HTTPServer((args.host, args.port), ChatHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
