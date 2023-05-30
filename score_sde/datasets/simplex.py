import os
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import softmax, log_softmax
from score_sde.datasets import TensorDataset


class DirichletMixture:
    def __init__(
        self,
        scale,
        scale_type,
        K,
        batch_dims,
        dim,
        seed,
        **kwargs
    ):
        self.K = K  # Number of mixture components
        self.batch_dims = batch_dims
        rng = jax.random.PRNGKey(seed)
        rng, next_rng = jax.random.split(rng)
        self.rng = rng
        if scale_type == "gamma_a":
            self.alphas = jax.random.gamma(key=next_rng, a=scale, shape=(K, dim))
        elif scale_type == "gamma_b":
            self.alphas = jax.random.gamma(key=next_rng, a=1., shape=(K, dim)) * scale
        elif scale_type == "fixed":
            self.alphas = jnp.ones((K, dim)) * scale
        else:
            raise NotImplementedError
        self.weights = jnp.ones(K) / K

    def __iter__(self):
        return self

    def __next__(self):
        return (self.sample(self.batch_dims), None)

    def sample(self, shape):
        rng = jax.random.split(self.rng, num=3)

        self.rng = rng[0]
        choice_key = rng[1]
        normal_key = rng[2]

        indices = jax.random.choice(
            choice_key, a=np.arange(self.K), shape=shape, p=self.weights
        )
        samples = jax.random.dirichlet(normal_key, self.alphas[indices], shape=shape)

        return samples


class ImageNetLatent(TensorDataset):
    def __init__(
        self,
        data_dir, 
        transform='softmax',
        dim=1000, 
        test=False,
        **kwargs
    ):
        data = np.load(os.path.join(data_dir, "imagenet_logits_100000.npy"))[:, :dim]
        if transform == 'softmax':
            data = softmax(data, axis=-1)
        elif transform == 'exp_log_softmax':
            data = np.exp(log_softmax(data, axis=-1))
        else:
            raise NotImplementedError
        super().__init__(jnp.array(data, dtype=jax.dtypes.canonicalize_dtype(float)))
