import importlib

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = ["Computer Modern Roman"]
# plt.rcParams.update({"font.size": 20})

from score_sde.datasets.conditional import GAndK
from score_sde.datasets.simplex import DirichletMixture, ImageNetLatent
from score_sde.models.distribution import DirichletDistribution

import jax
from jax import numpy as jnp
import numpy as np
from scipy.stats import norm, beta, dirichlet, entropy
import pandas as pd

try:
    plt.switch_backend("MACOSX")
except ImportError as error:
    plt.switch_backend("agg")
import seaborn as sns


MAX_PLOT_DIMS = 5


def plot_gandk(x0s, xts, dataset=None, close=True, **kwargs):
    x0, xt = dataset.unnormalize_ABgk(x0s[0]), dataset.unnormalize_ABgk(xts[0])
    parameter_names = ["A", "B", "g", "k"]
    metrics_dict = {}
    if not dataset.test:
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        for j in range(4):
            ax = axs[j]
            sns.regplot(x=x0[:, j], y=xt[:, j], ax=ax, scatter_kws={'s': 5})
            ax.set_xlabel("Target " + parameter_names[j])
            ax.set_ylabel("Model " + parameter_names[j])
            mse = np.mean((x0[:, j] - xt[:, j])**2)
            ax.set_title("MSE: " + str(mse))
            metrics_dict["MSE_"+parameter_names[j]] = mse
    else:
        true_ABgk = dataset.test_true_ABgk
        df_results = pd.DataFrame(xt, columns=parameter_names)

        ax = sns.pairplot(df_results, diag_kind='kde', plot_kws={'s': 5})
        true_parameters = iter(true_ABgk[0])
        def diag(x, **kwargs):
            true_parameter = next(true_parameters)
            plt.axvline(true_parameter, ls='--')
            plt.title("MSE: " + str(np.mean((x - true_parameter)**2)))
        ax.map_diag(diag)
        fig = ax.figure
        
    if close:
        plt.close(fig)
    return fig, metrics_dict


def plot_simplex(x0s, xts, size=10, dataset=None, close=True, **kwargs):
    x0, xt = x0s[0], xts[0]
    metrics_dict = {}

    dim = min(xt.shape[-1], MAX_PLOT_DIMS)
    colors = sns.color_palette("husl", len(xt))
    fig, axes = plt.subplots(
        2,
        dim,
        # figsize=(1.2 * size, 0.6 * size),
        figsize=(2 * size, size),
        sharex=False,
        sharey=True,
        tight_layout=True,
    )
    bins = 50
    for i, x, label in zip([0, 1], [x0, xt], ['data', 'model']):
        w = np.array(x)
        for j in range(dim):
            axes[i, j].hist(
                w[:, j],
                bins=bins,
                density=isinstance(dataset, DirichletMixture),
                alpha=0.3,
                color=colors[0],
                label=r"$x_{t_0}$" + ' ' + label,
            )
            axes[i, j].set(xlim=(0, 1))
            axes[i, j].set_xlabel(rf"$e_{j+1}$", fontsize=30)
            axes[i, j].tick_params(axis="both", which="major", labelsize=20)
            if isinstance(dataset, DirichletMixture):
                grid = np.linspace(0, 1, 100)
                y = np.sum([beta.pdf(grid, dataset.alphas[i][j], dataset.alphas[i].sum() - dataset.alphas[i][j]) * dataset.weights[i] for i in range(dataset.K)], axis=0)
                axes[i, j].plot(grid, y, alpha=0.5, lw=4, color="black", label=r"$p_{data}$")
            if j == 0:
                axes[i, j].legend(loc="best", fontsize=20)

        if isinstance(dataset, DirichletMixture) and label == 'data':
            vpdf = jax.vmap(jax.scipy.stats.dirichlet.pdf, (0, None), 0)
            metrics_dict[f'loglik_{label}'] = np.mean(np.log(np.sum([vpdf(w, dataset.alphas[i]) * dataset.weights[i] for i in range(dataset.K)], axis=0)))
        
        entropy_w = entropy(w, axis=-1)
        metrics_dict[f'entropy_mean_{label}'] = np.mean(entropy_w)
        metrics_dict[f'entropy_std_{label}'] = np.std(entropy_w)
    
    print(metrics_dict)

    if close:
        plt.close(fig)
    return fig, metrics_dict


def plot_normal(x, size=10, close=True):
    dim = min(x.shape[-1], MAX_PLOT_DIMS)
    colors = sns.color_palette("husl", len(x))
    fig, axes = plt.subplots(
        1,
        dim,
        # figsize=(1.2 * size, 0.6 * size),
        figsize=(2 * size, 0.5 * size),
        sharex=False,
        sharey=True,
        tight_layout=True,
    )
    bins = 100
    w = np.array(x)
    for j in range(dim):
        grid = np.linspace(-3, 3, 100)
        y = norm().pdf(grid)
        axes[j].hist(
            w[:, j],
            bins=bins,
            density=True,
            alpha=0.3,
            color=colors[0],
            label=r"$x_{t_f}$",
        )
        axes[j].set(xlim=(grid[0], grid[-1]))
        axes[j].set_xlabel(rf"$e_{j+1}$", fontsize=30)
        axes[j].tick_params(axis="both", which="major", labelsize=20)
        axes[j].plot(grid, y, alpha=0.5, lw=4, color="black", label=r"$p_{ref}$")
        if j == 0:
            axes[j].legend(loc="best", fontsize=20)
    
    if close:
        plt.close(fig)
    return fig, {}


def plot_dirichlet_base(x, size=10, base=None, close=True):
    dim = min(x.shape[-1], MAX_PLOT_DIMS)
    metrics_dict = {}
    colors = sns.color_palette("husl", len(x))
    fig, axes = plt.subplots(
        1,
        dim,
        # figsize=(1.2 * size, 0.6 * size),
        figsize=(2 * size, 0.5 * size),
        sharex=True,
        sharey=True,
        tight_layout=True,
    )
    bins = 50
    w = np.array(x)
    for j in range(dim):
        axes[j].hist(
            w[:, j],
            bins=bins,
            density=True,
            alpha=0.3,
            color=colors[0],
            label=r"$x_{t_f}$",
        )
        axes[j].set_xlabel(rf"$e_{j+1}$", fontsize=30)
        axes[j].tick_params(axis="both", which="major", labelsize=20)
        grid = np.linspace(0, 1, 100)
        y = beta.pdf(grid, base.alpha[j], base.alpha.sum() - base.alpha[j])
        if np.max(y) <= 20:
            axes[j].set(xlim=(grid[0], grid[-1]))
            axes[j].plot(grid, y, alpha=0.5, lw=4, color="black", label=r"$p_{ref}$")
        if j == 0:
            axes[j].legend(loc="best", fontsize=20)
    
    metrics_dict['loglik_pT_model'] = np.mean(base.log_prob(x))
    metrics_dict['loglik_pT_data'] = - base.entropy()

    print(metrics_dict)

    if close:
        plt.close(fig)
    return fig, metrics_dict


def plot(x0, xt, dataset=None, prob=None, size=10, close=True):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    if isinstance(dataset, GAndK):
        fig, metrics_dict = plot_gandk(x0, xt, dataset=dataset, close=close)
    elif isinstance(dataset, DirichletMixture) or isinstance(dataset, ImageNetLatent):
        fig, metrics_dict = plot_simplex(x0, xt, dataset=dataset, close=close)
    else:
        fig, metrics_dict = None, {}
    return fig, metrics_dict


def plot_ref(xt, size=10, base=None, close=True):
    if isinstance(base, DirichletDistribution):
        fig, metrics_dict = plot_dirichlet_base(xt, size, base=base, close=close)
    else:
        fig, metrics_dict = plot_normal(xt, size, close=close)
    return fig, metrics_dict
