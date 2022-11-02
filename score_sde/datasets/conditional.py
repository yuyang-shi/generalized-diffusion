from functools import partial
from math import prod, floor
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.stats import norm
from score_sde.datasets import TensorDataset

def _g_and_k_transform(z, ABgk):
    # ABgk: shape (batch_size, 4)
    A, B, g, k = ABgk[:, 0:1], ABgk[:, 1:2], ABgk[:, 2:3], ABgk[:, 3:4]
    return A + B * (1 + 0.8 * (1 - jnp.exp(-g * z)) / (1 + jnp.exp(-g * z))) * jnp.power((1 + z**2), k) * z


class GAndK(TensorDataset):
    def __init__(
        self,
        rng,
        num_data, 
        num_samples, 
        num_quantiles, 
        basis_degree=4, 
        normalize_type="meanstd",  # "minmax"
        prior_range=[0,10],
        test=False,
        **kwargs
    ):  
        assert num_quantiles <= num_samples
        self.test_true_ABgk = jnp.array([[3, 1, 2, 0.5]])
        self.num_samples = num_samples
        self.prior_range = prior_range
        self.test = test
        quantile_every = num_samples // num_quantiles
        # self.quantile_idx = jnp.sort(jnp.unique(jnp.arange(-1, num_samples, quantile_every).clip(0)))
        self.quantile_idx = jnp.sort(jnp.unique(jnp.arange(quantile_every-1, num_samples-1, quantile_every).clip(0)))
        self.quantiles = self.quantile_idx / (num_samples - 1)

        self.rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)

        if not self.test:
            ABgk = jax.random.uniform(rng1, shape=[num_data, 4], minval=self.prior_range[0], maxval=self.prior_range[1])
        else:
            ABgk = jnp.broadcast_to(self.test_true_ABgk, shape=[num_data, 4])

        # Gamma exponential clever sampling
        if num_samples > 250:
            spacing = jnp.concatenate([jnp.array([self.quantile_idx[0]+1]), self.quantile_idx[1:] - self.quantile_idx[:-1]])
            batch_size = 200000
            sample_quantiles = []
            for i in range(int(np.ceil(num_data / batch_size))):
                current_batch_size = min(batch_size, num_data - i*batch_size)
                rng2, next_rng2 = jax.random.split(rng2)
                gam = jax.random.gamma(next_rng2, jnp.expand_dims(spacing, 1), (len(spacing),current_batch_size))
                rng3, next_rng3 = jax.random.split(rng3)
                exp = jax.random.gamma(next_rng3, num_samples - self.quantile_idx[-1], (1,current_batch_size))
                cumgam = jnp.cumsum(gam, 0)
                u_quantiles = cumgam / (jnp.sum(gam, 0) + exp)
                n_quantiles = norm.ppf(u_quantiles).T
                sample_quantiles.append(_g_and_k_transform(n_quantiles, ABgk[(i*batch_size):(i*batch_size+current_batch_size)]))
            sample_quantiles = jnp.concatenate(sample_quantiles)

        else:
            # Ordinary sampling
            normal_samples = jax.random.normal(rng2, shape=[num_data, num_samples])
            samples = _g_and_k_transform(normal_samples, ABgk)
            # sample_quantiles = jnp.quantile(samples, self.quantiles, axis=1).T
            sample_quantiles = jnp.sort(samples, axis=1)[:, self.quantile_idx]

        # pd.DataFrame(sample_quantiles).to_csv('sample_quantiles.csv', header=False) 

        ABgk_samples = self.normalize_ABgk(ABgk)

        if normalize_type == "meanstd":
            sample_quantiles = jnp.concatenate([sample_quantiles**d for d in range(1, basis_degree+1)], 1)
            sample_quantiles_mean = jnp.mean(sample_quantiles, 0)
            sample_quantiles_std = jnp.std(sample_quantiles, 0)
            sample_quantiles = (sample_quantiles - sample_quantiles_mean) / sample_quantiles_std
        
        elif normalize_type == "minmax":
            sample_quantiles = jnp.concatenate([sample_quantiles**d for d in range(1, basis_degree+1)], 1)
            sample_quantiles_min = jnp.min(sample_quantiles, 0)
            sample_quantiles_max = jnp.max(sample_quantiles, 0)
            sample_quantiles_mean = 1/2 * (sample_quantiles_min + sample_quantiles_max)
            sample_quantiles_std = 1/2 * (sample_quantiles_max - sample_quantiles_min)
            sample_quantiles = (sample_quantiles - sample_quantiles_mean) / sample_quantiles_std

        elif normalize_type == "premeanstd":
            sample_quantiles_mean = jnp.mean(sample_quantiles, 0)
            sample_quantiles_std = jnp.std(sample_quantiles, 0)
            sample_quantiles = (sample_quantiles - sample_quantiles_mean) / sample_quantiles_std
            sample_quantiles = jnp.concatenate([sample_quantiles**d for d in range(1, basis_degree+1)], 1)

        jnp.savez("sample_quantiles_normalization.npz", mean=sample_quantiles_mean, std=sample_quantiles_std)

        super().__init__(ABgk_samples, sample_quantiles)
    
    def normalize_ABgk(self, ABgk):
        # normalized_ABgk = (ABgk - self.prior_range[0]) / (self.prior_range[1] - self.prior_range[0])
        # return normalized_ABgk*2 - 1
        return ABgk
    
    def unnormalize_ABgk(self, normalized_ABgk):
        # normalized_ABgk = (normalized_ABgk + 1) / 2
        # return jnp.clip(normalized_ABgk * (self.prior_range[1] - self.prior_range[0]) + self.prior_range[0], self.prior_range[0], self.prior_range[1])
        return jnp.clip(normalized_ABgk, self.prior_range[0], self.prior_range[1])
        
    def simulate_quantiles(self, ABgk):
        self.rng, next_rng = jax.random.split(self.rng, num=2)

        normal_samples = jax.random.normal(next_rng, shape=[ABgk.shape[0], self.num_samples])
        
        samples = _g_and_k_transform(normal_samples, ABgk)
        # sample_quantiles = jnp.quantile(samples, self.quantiles, axis=1).T
        sample_quantiles = jnp.sort(samples, axis=1)[:, self.quantile_idx]
        return sample_quantiles

    def quantiles_to_context(self, sample_quantiles, sample_quantiles_normalization):
        sample_quantiles_mean = sample_quantiles_normalization['mean']
        sample_quantiles_std = sample_quantiles_normalization['std']

        if normalize_type == "meanstd":
            sample_quantiles = jnp.concatenate([sample_quantiles**d for d in range(1, basis_degree+1)], 1)
            sample_quantiles = (sample_quantiles - sample_quantiles_mean) / sample_quantiles_std
        
        elif normalize_type == "minmax":
            sample_quantiles = jnp.concatenate([sample_quantiles**d for d in range(1, basis_degree+1)], 1)
            sample_quantiles = (sample_quantiles - sample_quantiles_mean) / sample_quantiles_std

        elif normalize_type == "premeanstd":
            sample_quantiles = (sample_quantiles - sample_quantiles_mean) / sample_quantiles_std
            sample_quantiles = jnp.concatenate([sample_quantiles**d for d in range(1, basis_degree+1)], 1)
        
        return sample_quantiles

if __name__ == "__main__":    
    rng = jax.random.PRNGKey(2)
    rng, next_rng = jax.random.split(rng)
    gandk = GAndK(next_rng, 1050000, 10000, 250, basis_degree=1, normalize_type="premeanstd", prior_range=[0, 10])
    print(gandk.quantile_idx)
    print(gandk.context_data.mean(0))
    print(gandk.context_data.std(0))

    raise NotImplementedError
    ABgk = jnp.array([[3, 1, 2, 0.5]])
    rng, next_rng = jax.random.split(rng)
    normal_samples = jax.random.normal(next_rng, shape=[1, 10000])
    samples = _g_and_k_transform(normal_samples, ABgk)

    from matplotlib import pyplot as plt
    from scipy.stats import norm
    import seaborn as sns
    lin = np.linspace(0, 1, 1000)
    plt.plot(lin, np.quantile(jnp.squeeze(samples), lin), label="sample")
    plt.plot(lin, np.squeeze(_g_and_k_transform(np.expand_dims(norm.ppf(lin), 0), ABgk)), label="theoretical")
    # plt.scatter(np.repeat(np.expand_dims(np.arange(1, 100) / 100, 0), 1000, 0).reshape(-1), gandk.context_data.reshape(-1))
    plt.legend()
    plt.savefig("test.png")

    # sns.kdeplot(jnp.squeeze(samples))
    # plt.xlim((-5, 15))
    # plt.savefig("test.png")




# class GAndK:
#     def __init__(
#         self,
#         batch_dims,
#         rng,
#         num_samples, 
#         num_quantiles, 
#         basis_degree=4, 
#         prior_range=[0,10],
#         test=False,
#         **kwargs
#     ):  
#         assert num_quantiles <= num_samples
#         self.num_samples = num_samples
#         self.num_quantiles = num_quantiles
#         self.basis_degree = basis_degree
#         self.prior_range = prior_range
#         self.batch_dims = batch_dims
#         self.rng = rng
#         self.test = test
#         self.test_true_ABgk = jnp.array([[3, 1, 2, 0.5]])
#         self.quantiles = jnp.arange(1, num_quantiles) / num_quantiles

#     def __iter__(self):
#         return self

#     def __next__(self):
#         self.rng, rng1, rng2 = jax.random.split(self.rng, num=3)

#         if not self.test:
#             ABgk = jax.random.uniform(rng1, shape=self.batch_dims + [4], minval=self.prior_range[0], maxval=self.prior_range[1])
#         else:
#             ABgk = jnp.broadcast_to(self.test_true_ABgk, shape=self.batch_dims + [4])
        
#         normal_samples = jax.random.normal(rng2, shape=self.batch_dims + [self.num_samples])
        
#         samples = _g_and_k_transform(normal_samples, ABgk)
#         sample_quantiles = jnp.quantile(samples, self.quantiles, axis=1)
#         sample_quantiles = jnp.concatenate([sample_quantiles**d for d in range(1, self.basis_degree+1)], 0)
#         return self.normalize_ABgk(ABgk), sample_quantiles.T
    
#     def normalize_ABgk(self, ABgk):
#         normalized_ABgk = (ABgk - self.prior_range[0]) / (self.prior_range[1] - self.prior_range[0])
#         return normalized_ABgk*2 - 1
    
#     def unnormalize_ABgk(self, normalized_ABgk):
#         normalized_ABgk = (normalized_ABgk + 1) / 2
#         return jnp.clip(normalized_ABgk * (self.prior_range[1] - self.prior_range[0]) + self.prior_range[0], self.prior_range[0], self.prior_range[1])

#     def simulate(self, ABgk):
#         self.rng, rng = jax.random.split(self.rng, num=2)

#         normal_samples = jax.random.normal(rng, shape=[ABgk.shape[0], self.num_samples])
        
#         samples = _g_and_k_transform(normal_samples, ABgk)
#         sample_quantiles = jnp.quantile(samples, self.quantiles, axis=1)
#         return sample_quantiles.T