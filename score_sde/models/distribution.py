import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats


class NormalDistribution:
    def __init__(self, **kwargs):
        pass

    def sample(self, rng, shape):
        return jax.random.normal(rng, shape)

    def log_prob(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logp_fn = lambda z: -N / 2.0 * jnp.log(2 * np.pi) - jnp.sum(z**2) / 2.0
        return jax.vmap(logp_fn)(z)


class DirichletDistribution:
    def __init__(self, alpha, dim, **kwargs):
        self.alpha = jnp.ones([dim]) * alpha 
        self.dim = dim
    
    def sample(self, rng, shape):
        return jax.random.dirichlet(rng, self.alpha, shape[:-1])

    def log_prob(self, z):
        vlogpdf = jax.vmap(jax.scipy.stats.dirichlet.logpdf, (0, None), 0)
        return vlogpdf(z, self.alpha)

    def entropy(self):
        return scipy.stats.dirichlet.entropy(self.alpha)
