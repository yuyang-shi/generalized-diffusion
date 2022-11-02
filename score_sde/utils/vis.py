import importlib

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = ["Computer Modern Roman"]
# plt.rcParams.update({"font.size": 20})

from score_sde.datasets.conditional import GAndK

import jax
from jax import numpy as jnp
import numpy as np
from scipy.stats import norm
import pandas as pd

try:
    plt.switch_backend("MACOSX")
except ImportError as error:
    plt.switch_backend("agg")
import seaborn as sns

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError as error:
    pass


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

def plot_normal(x, size=10, close=True):
    dim = x.shape[-1]
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
    for j in range(w.shape[-1]):
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


def plot(x0, xt, dataset=None, prob=None, size=10, close=True):
    if isinstance(dataset.dataset.dataset, GAndK):
        fig, metrics_dict = plot_gandk(x0, xt, dataset=dataset.dataset.dataset, close=close)
    return fig, metrics_dict


def plot_ref(xt, size=10, close=True):
    fig, metrics_dict = plot_normal(xt, size, close=close)
    return fig, metrics_dict
