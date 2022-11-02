import jax
import jax.numpy as jnp

from score_sde.sde import SDE, VPSDE as VPSDEBase, RSDE as RSDEBase
from score_sde.utils import batch_mul
from riemannian_score_sde.sampling import get_pc_sampler
from riemannian_score_sde.models.distribution import UniformDistribution


class Brownian(SDE):
    def __init__(self, manifold, tf: float, t0: float = 0, beta_0=0.1, beta_f=20, N=100):
        """Construct a Brownian motion on a compact manifold"""
        super().__init__(tf, t0)
        self.beta_0 = beta_0
        self.beta_f = beta_f
        self.manifold = manifold
        self.limiting = UniformDistribution(manifold)
        self.N = N

    def beta_t(self, t):
        normed_t = (t - self.t0) / (self.tf - self.t0)
        return self.beta_0 + normed_t * (self.beta_f - self.beta_0)

    def rescale_t(self, t):
        return 0.5 * t**2 * (self.beta_f - self.beta_0) + t * self.beta_0

    def coefficients(self, x, t):
        beta_t = self.beta_t(t)
        drift = jnp.zeros_like(x)
        diffusion = jnp.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """Should not rely on closed-form marginal probability"""
        # NOTE: this is a Euclidean approx!
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_f - self.beta_0) - 0.5 * t * self.beta_0
        )
        # mean = batch_mul(jnp.exp(log_mean_coeff), x)
        std = jnp.sqrt(1 - jnp.exp(2.0 * log_mean_coeff))
        return jnp.zeros_like(x), std

    def marginal_sample(self, rng, x, t, return_hist=False, fast_sampling=True):
        if return_hist or (not fast_sampling):
            out = None
        else:
            out = self.manifold.random_walk(rng, x, self.rescale_t(t))
        if out is None:
            sampler = get_pc_sampler(
                self,
                self.N,
                predictor="GRW",
                return_hist=return_hist,
            )
            out = sampler(rng, x, tf=t)
        return out

    def grad_marginal_log_prob(self, x0, x, t, **kwargs):
        s = self.rescale_t(t)
        logp_grad = self.manifold.grad_marginal_log_prob(x0, x, s, **kwargs)
        return None, logp_grad

    def varhadan_exp(self, xs, xt, s, t):
        delta_t = self.rescale_t(t) - self.rescale_t(s)
        axis_to_expand = tuple(range(-1, -len(xt.shape), -1))  # (-1) or (-1, -2)
        delta_t = jnp.expand_dims(delta_t, axis=axis_to_expand)
        grad = self.manifold.log(xs, xt) / delta_t
        return delta_t, grad

    def sample_limiting_distribution(self, rng, shape):
        return self.limiting.sample(rng, shape)

    def limiting_distribution_logp(self, z):
        return self.limiting.log_prob(z)

    def reverse(self, score_fn):
        return RSDE(self, score_fn)


class VPSDE(VPSDEBase):
    def __init__(self, tf: float, t0: float = 0, beta_0=0.1, beta_f=20, manifold=None):
        super().__init__(tf, t0, beta_0, beta_f)
        self.manifold = manifold

    def rescale_t(self, t):
        return 0.5 * t**2 * (self.beta_f - self.beta_0) + t * self.beta_0

    def marginal_sample(self, rng, x, t):
        mean, std = self.marginal_prob(x, t)
        z = jax.random.normal(rng, x.shape)
        return mean + batch_mul(std, z)

    def grad_marginal_log_prob(self, x0, x, t, **kwargs):
        mean, std = self.marginal_prob(x0, t)
        std = jnp.expand_dims(std, -1)
        score = -1 / (std**2) * (x - mean)
        logp = None
        return logp, score

    def reverse(self, score_fn):
        return RSDE(self, score_fn)


class RSDE(RSDEBase):
    def __init__(self, sde: SDE, score_fn):
        super().__init__(sde, score_fn)
        self.manifold = sde.manifold
