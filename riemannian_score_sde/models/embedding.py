import abc

import jax
import haiku as hk
import jax.numpy as jnp


class Embedding(hk.Module, abc.ABC):
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold


class NoneEmbedding(Embedding):
    def __call__(self, x, t):
        return x, t
