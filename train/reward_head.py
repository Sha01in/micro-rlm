"""Flax NNX reward head module for trajectory scoring.

Maps the last hidden state of a language model backbone to a scalar
reward in [0, 1]. Used by reward_model.py to score RLM trajectory quality.

Architecture: single linear projection → sigmoid.
Trained from scratch while the backbone is fine-tuned with LoRA.
"""

from flax import nnx
import jax
import jax.numpy as jnp


class RewardHead(nnx.Module):
    """Reward head: hidden_dim → Linear(1) → sigmoid → [0, 1] scalar.

    Args:
        hidden_dim: Backbone's hidden size (e.g., 1536 for Qwen2.5-1.5B).
        rngs: Flax NNX RNG container for parameter initialization.
    """

    def __init__(self, hidden_dim: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, last_hidden_state: jax.Array) -> jax.Array:
        """Score a position's hidden state.

        Args:
            last_hidden_state: Shape (batch, hidden_dim) or (hidden_dim,).
        Returns:
            Reward score in [0, 1]. Shape (batch,) or scalar.
        """
        return jax.nn.sigmoid(self.linear(last_hidden_state)).squeeze(-1)
