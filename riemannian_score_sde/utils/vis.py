import os
os.environ['GEOMSTATS_BACKEND'] = 'jax'

import math
import importlib

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = ["Computer Modern Roman"]
# plt.rcParams.update({"font.size": 20})

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.special_orthogonal import (
    _SpecialOrthogonalMatrices,
    _SpecialOrthogonal3Vectors,
)
from geomstats.geometry.product_manifold import ProductSameManifold

import jax
from jax import numpy as jnp
import numpy as np
from scipy.stats import norm

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


def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    zmean = np.mean(zlim)

    plot_radius = max(
        [
            abs(lim - mean_)
            for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
            for lim in lims
        ]
    )

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

    return ax


def get_sphere_coords():
    radius = 1.0
    # set_aspect_equal_3d(ax)
    n = 200
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    return x, y, z


def sphere_plot(ax, color="grey"):
    # assert manifold.dim == 2
    x, y, z = get_sphere_coords()
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, linewidth=0, alpha=0.2)

    return ax

    # ax.set_xlim3d(-radius, radius)
    # ax.set_ylim3d(-radius, radius)
    # ax.set_zlim3d(-radius, radius)


def remove_background(ax):
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return ax


def latlon_from_cartesian(points):
    r = jnp.linalg.norm(points, axis=-1)
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    lat = -jnp.arcsin(z / r)
    lon = jnp.arctan2(y, x)
    # lon = jnp.where(lon > 0, lon - math.pi, lon + math.pi)
    return jnp.concatenate([jnp.expand_dims(lat, -1), jnp.expand_dims(lon, -1)], axis=-1)


def cartesian_from_latlong(points):
    lat = points[..., 0]
    lon = points[..., 1]

    x = jnp.cos(lat) * jnp.cos(lon)
    y = jnp.cos(lat) * jnp.sin(lon)
    z = jnp.sin(lat)

    return jnp.stack([x, y, z], axis=-1)


def get_spherical_grid(N, eps=0.0):
    lat = jnp.linspace(-90 + eps, 90 - eps, N // 2)
    lon = jnp.linspace(-180 + eps, 180 - eps, N)
    Lat, Lon = jnp.meshgrid(lat, lon)
    latlon_xs = jnp.concatenate([Lat.reshape(-1, 1), Lon.reshape(-1, 1)], axis=-1)
    spherical_xs = jnp.pi * (latlon_xs / 180.0) + jnp.array([jnp.pi / 2, jnp.pi])[None, :]
    xs = Hypersphere(2).spherical_to_extrinsic(spherical_xs)
    return xs, lat, lon


def plot_3d(x0s, xts, size, prob):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    ax = remove_background(ax)
    fig.subplots_adjust(left=-0.2, bottom=-0.2, right=1.2, top=1.2, wspace=0, hspace=0)
    # ax.view_init(elev=30, azim=45)
    ax.view_init(elev=0, azim=0)
    cmap = sns.cubehelix_palette(as_cmap=True)
    sphere = visualization.Sphere()
    sphere.draw(ax, color="red", marker=".")
    # sphere_plot(ax)
    # sphere.plot_heatmap(ax, pdf, n_points=16000, alpha=0.2, cmap=cmap)
    for k, (x0, xt) in enumerate(zip(x0s, xts)):
        if x0 is not None:
            cax = ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], s=50, color="green")
        if xt is not None:
            x, y, z = xt[:, 0], xt[:, 1], xt[:, 2]
            c = prob if prob is not None else np.ones([*xt.shape[:-1]])
            cax = ax.scatter(x, y, z, s=50, vmin=0.0, vmax=2.0, c=c, cmap=cmap)
        # if grad is not None:
        #     u, v, w = grad[:, 0], grad[:, 1], grad[:, 2]
        #     quiver = ax.quiver(
        #         x, y, z, u, v, w, length=0.2, lw=2, normalize=False, cmap=cmap
        #     )
        #     quiver.set_array(c)

    plt.colorbar(cax)
    # plt.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return fig


def earth_plot(cfg, log_prob, train_ds, test_ds, N, azimuth=None, samples=None):
    """generate earth plots with model density or integral paths aka streamplot"""
    has_cartopy = importlib.find_loader("cartopy")
    print("has_cartopy", has_cartopy)
    if not has_cartopy:
        return

    # parameters
    azimuth_dict = {"earthquake": 70, "fire": 50, "floow": 60, "volcanoe": 170}
    azimuth = azimuth_dict[str(cfg.dataset.name)] if azimuth is None else azimuth
    polar = 30
    # projs = ["ortho", "robinson"]
    projs = ["ortho"]

    xs, lat, lon = get_spherical_grid(N, eps=0.0)
    # ts = [0.01, 0.05, cfg.flow.tf]
    ts = [cfg.flow.tf]
    figs = []

    for t in ts:
        print(t)
        # fs = log_prob(xs, t)
        fs = log_prob(xs)
        fs = fs.reshape((lat.shape[0], lon.shape[0]), order="F")
        fs = jnp.exp(fs)
        # norm = mcolors.PowerNorm(3.)  # NOTE: tweak that value
        norm = mcolors.PowerNorm(0.2)  # N=500
        fs = np.array(fs)
        # print(np.min(fs).item(), jnp.quantile(fs, np.array([0.1, 0.5, 0.9])), np.max(fs).item())
        fs = norm(fs)
        # print(np.min(fs).item(), jnp.quantile(fs, np.array([0.1, 0.5, 0.9])), np.max(fs).item())

        # create figure with earth features
        for i, proj in enumerate(projs):
            print(proj)
            fig = plt.figure(figsize=(5, 5), dpi=300)
            if proj == "ortho":
                projection = ccrs.Orthographic(azimuth, polar)
            elif proj == "robinson":
                projection = ccrs.Robinson(central_longitude=0)
            else:
                raise Exception("Invalid proj {}".format(proj))
            ax = fig.add_subplot(1, 1, 1, projection=projection, frameon=True)
            ax.set_global()

            # earth features
            ax.add_feature(cfeature.LAND, zorder=0, facecolor="#e0e0e0")

            vmin, vmax = 0.0, 1.0
            # n_levels = 900
            n_levels = 200
            levels = np.linspace(vmin, vmax, n_levels)
            cmap = sns.cubehelix_palette(
                light=1.0, dark=0.0, start=0.5, rot=-0.75, reverse=False, as_cmap=True
            )
            # cmap = sns.cubehelix_palette(as_cmap=True)
            cs = ax.contourf(
                lon,
                lat,
                fs,
                levels=levels,
                # alpha=0.8,
                transform=ccrs.PlateCarree(),
                antialiased=True,
                # vmin=vmin,
                # vmax=vmax,
                cmap=cmap,
                extend="both",
            )

            alpha_gradient = np.linspace(0, 1, len(ax.collections))
            for col, alpha in zip(ax.collections, alpha_gradient):
                col.set_alpha(alpha)
            # for col in ax.collections[0:1]:
            # col.set_alpha(0)

            # add scatter plots of the dataset
            colors = sns.color_palette("hls", 8)
            # colors = sns.color_palette()
            train_idx = train_ds.dataset.indices
            test_idx = test_ds.dataset.indices
            if samples is not None:
                samples = np.array(latlon_from_cartesian(samples)) * 180 / math.pi
                points = projection.transform_points(
                    ccrs.Geodetic(), samples[:, 1], samples[:, 0]
                )
                ax.scatter(points[:, 0], points[:, 1], s=1.0, c=[colors[1]], alpha=1.0)
            samples = train_ds.dataset.dataset.data
            samples = np.array(latlon_from_cartesian(samples)) * 180 / math.pi
            points = projection.transform_points(
                ccrs.Geodetic(), samples[:, 1], samples[:, 0]
            )
            ax.scatter(
                points[train_idx, 0],
                points[train_idx, 1],
                s=0.2,
                c=[colors[5]],
                alpha=0.2,
            )
            ax.scatter(
                points[test_idx, 0],
                points[test_idx, 1],
                s=0.2,
                c=[colors[0]],
                alpha=0.2,
            )
            # plt.close(fig)
            figs.append(fig)

    return figs


def plot_so3(x0s, xts, size, close=True, **kwargs):
    colors = sns.color_palette("husl", len(x0s))
    # colors = sns.color_palette("tab10")
    fig, axes = plt.subplots(
        2,
        3,
        # figsize=(1.2 * size, 0.6 * size),
        figsize=(2 * size, 1 * size),
        sharex=False,
        sharey="col",
        tight_layout=True,
    )
    # x_labels = [r"$\alpha$", r"$\beta$", r"$\gamma$"]
    x_labels = [r"$\phi$", r"$\theta$", r"$\psi$"]
    y_labels = ["Ground truth", "Model"]
    # bins = round(math.sqrt(len(w[:, 0])))
    bins = 100

    for k, (x0, xt) in enumerate(zip(x0s, xts)):
        # print(k, x0.shape, xt.shape)
        for i, x in enumerate([x0, xt]):
            w = _SpecialOrthogonal3Vectors().tait_bryan_angles_from_matrix(x)
            # w = _SpecialOrthogonal3Vectors().rotation_vector_from_matrix(x)
            w = np.array(w)
            for j in range(3):
                axes[i, j].hist(
                    w[:, j],
                    bins=bins,
                    density=True,
                    alpha=0.3,
                    color=colors[k],
                    label=f"Component #{k}",
                )
                if j == 1:
                    axes[i, j].set(xlim=(-math.pi / 2, math.pi / 2))
                    axes[i, j].set_xticks([-math.pi / 2, 0, math.pi / 2])
                    axes[i, j].set_xticklabels([r"$-\pi/2$", "0", r"$\pi/2$"], color="k")
                else:
                    axes[i, j].set(xlim=(-math.pi, math.pi))
                    axes[i, j].set_xticks([-math.pi, 0, math.pi])
                    axes[i, j].set_xticklabels([r"$-\pi$", "0", r"$\pi$"], color="k")
                if j == 0:
                    axes[i, j].set_ylabel(y_labels[i], fontsize=30)
                # if i == 0 and j == 0:
                # axes[i, j].legend(loc="best", fontsize=20)
                if i == 0:
                    axes[i, j].get_xaxis().set_visible(False)
                if i == 1:
                    axes[i, j].set_xlabel(x_labels[j], fontsize=30)
                axes[i, j].tick_params(axis="both", which="major", labelsize=20)

    if close:
        plt.close(fig)
    return fig


def proj_t2(x):
    return jnp.mod(
        jnp.stack(
            [jnp.arctan2(x[..., 0], x[..., 1]), jnp.arctan2(x[..., 2], x[..., 3])],
            axis=-1,
        ),
        jnp.pi * 2,
    )


def plot_t2(x0s, xts, size, **kwargs):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(0.6 * size, 0.6 * size),
        sharex=False,
        sharey=False,
        tight_layout=True,
    )

    for k, (x0, xt) in enumerate(zip(x0s, xts)):
        for i, x in enumerate([x0, xt]):
            x = proj_t2(x)
            axes[i].scatter(x[..., 0], x[..., 1], s=0.1)

    for ax in axes:
        ax.set_xlim([0, 2 * jnp.pi])
        ax.set_ylim([0, 2 * jnp.pi])
        ax.set_aspect("equal")

    plt.close(fig)
    return fig


import seaborn as sns


def plot_tn(x0s, xts, size, **kwargs):
    n = x0s[0].shape[-1]
    n = min(5, n // 4)

    fig, axes = plt.subplots(
        n,
        2,
        figsize=(0.6 * size, 0.6 * size * n / 2),
        sharex=False,
        sharey=False,
        tight_layout=True,
        squeeze=False,
    )
    # cmap = sns.mpl_palette("viridis")
    for k, (x0, xt) in enumerate(zip(x0s, xts)):
        print(x0.shape)
        for i, x in enumerate([x0, xt]):
            for j in range(n):
                x_ = proj_t2(x[..., (4 * j) : (4 * (j + 1))])
                axes[j][i].scatter(x_[..., 0], x_[..., 1], s=0.1)
                # sns.kdeplot(
                #     x=np.asarray(x_[..., 0]),
                #     y=np.asarray(x_[..., 1]),
                #     ax=axes[j][i],
                #     cmap=cmap,
                #     fill=True,
                #     # levels=15,
                # )

    axes = [item for sublist in axes for item in sublist]
    for ax in axes:
        ax.set_xlim([0, 2 * jnp.pi])
        ax.set_ylim([0, 2 * jnp.pi])
        ax.set_aspect("equal")

    plt.close(fig)
    return fig


def proj_t1(x):
    return jnp.mod(jnp.arctan2(x[..., 0], x[..., 1]), 2 * np.pi)


def plot_t1(x0s, xts, size, **kwargs):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(0.6 * size, 0.6 * size),
        sharex=False,
        sharey=True,
        tight_layout=True,
    )

    for k, (x0, xt) in enumerate(zip(x0s, xts)):
        for i, x in enumerate([x0, xt]):
            x = proj_t1(x)
            sns.kdeplot(x, ax=axes[i])
            plt.scatter(jnp.zeros_like(x), x, marker="|")

    for ax in axes:
        ax.set_xlim([0, 2 * jnp.pi])

    plt.close(fig)
    return fig


def plot_so3_uniform(x, size=10, close=True):
    colors = sns.color_palette("husl", len(x))
    fig, axes = plt.subplots(
        1,
        3,
        # figsize=(1.2 * size, 0.6 * size),
        figsize=(2 * size, 0.5 * size),
        sharex=False,
        sharey=True,
        tight_layout=True,
    )
    x_labels = [r"$\phi$", r"$\theta$", r"$\psi$"]
    # bins = round(math.sqrt(len(w[:, 0])))
    bins = 100
    w = _SpecialOrthogonal3Vectors().tait_bryan_angles_from_matrix(x)
    w = np.array(w)

    for j in range(3):
        if j == 1:
            grid = np.linspace(-np.pi / 2, np.pi / 2, 100)
            axes[j].set_xticks([grid[0], 0, grid[-1]])
            axes[j].set_xticklabels([r"$-\pi/2$", "0", r"$\pi/2$"], color="k")
            y = np.sin(grid + math.pi / 2) / 2
        else:
            grid = np.linspace(-np.pi, np.pi, 100, endpoint=True)
            axes[j].set_xticks([grid[0], 0, grid[-1]])
            axes[j].set_xticklabels([r"$-\pi$", "0", r"$\pi$"], color="k")
            y = 1 / (2 * np.pi) * np.ones_like(grid)
        axes[j].hist(
            w[:, j],
            bins=bins,
            density=True,
            alpha=0.3,
            color=colors[0],
            label=r"$x_{t_f}$",
        )
        axes[j].set(xlim=(grid[0], grid[-1]))
        axes[j].set_xlabel(x_labels[j], fontsize=30)
        axes[j].tick_params(axis="both", which="major", labelsize=20)
        axes[j].plot(grid, y, alpha=0.5, lw=4, color="black", label=r"$p_{ref}$")
        if j == 0:
            axes[j].legend(loc="best", fontsize=20)
    if close:
        plt.close(fig)
    return fig


def plot_so3b(prob, lambda_x, N, size=10):
    fig, axes = plt.subplots(
        1,
        3,
        # figsize=(1.2 * size, 0.6 * size),
        figsize=(2 * size, 0.5 * size),
        sharex=False,
        sharey=True,
        tight_layout=True,
    )
    x_labels = [r"$\phi$", r"$\theta$", r"$\psi$"]
    prob = prob.reshape(N, N // 2, N)
    lambda_x = lambda_x.reshape(N, N // 2, N)

    for j in range(3):
        if j == 1:
            grid = np.linspace(-np.pi / 2, np.pi / 2, N // 2)
            axes[j].set_xticks([grid[0], 0, grid[-1]])
            axes[j].set_xticklabels([r"$-\pi/2$", "0", r"$\pi/2$"], color="k")
        else:
            grid = np.linspace(-np.pi, np.pi, N)
            axes[j].set_xticks([grid[0], 0, grid[-1]])
            axes[j].set_xticklabels([r"$-\pi$", "0", r"$\pi$"], color="k")

        y = jnp.mean(prob * lambda_x, axis=jnp.delete(jnp.arange(3), j))

        axes[j].set(xlim=(grid[0], grid[-1]))
        axes[j].set_xlabel(x_labels[j], fontsize=30)
        axes[j].tick_params(axis="both", which="major", labelsize=20)
        axes[j].plot(grid, y, alpha=0.5, lw=4, color="black", label=r"$p_{ref}$")
        if j == 0:
            axes[j].legend(loc="best", fontsize=20)

    plt.close(fig)
    return fig


def plot_so3c(x0s, xts, size, dataset=None, close=True, **kwargs):
    canonical_rotation=np.eye(3)
    show_color_wheel = True
    def _show_single_marker(ax, rotation, marker, edgecolors=True,
                            facecolors=False):
        w = _SpecialOrthogonal3Vectors().tait_bryan_angles_from_matrix(rotation, extrinsic_or_intrinsic="intrinsic", order="zyx")
        xyz = rotation[:, 0]
        tilt_angle = w[0]
        longitude = np.arctan2(xyz[0], -xyz[1])
        latitude = np.arcsin(xyz[2])

        color = cmap(0.5 + tilt_angle / 2 / np.pi)
        
        # phi = np.linspace(0, 2.*np.pi, 36)  #36 points
        # r = np.radians(40)
        # x = np.radians(35) + r*np.cos(phi)
        # y = np.radians(30) + r*np.sin(phi)
        # ax.plot(x, y, color="g")

        # ax.add_patch(plt.Circle((longitude, latitude), 0.3, color=color, linewidth=1, fill=False, alpha=0.2))
        # ax.scatter(longitude, latitude, s=2500,
        #            edgecolors=color if edgecolors else 'none',
        #            facecolors=facecolors if facecolors else 'none',
        #            marker=marker,
        #            linewidth=2)
        ax.scatter(longitude, latitude, s=75,
                edgecolors='black' if edgecolors else 'none',
                facecolors=color if facecolors else 'none',
                marker=marker,
                linewidth=1)

    fig = plt.figure(figsize=(16, 4), dpi=100)
    titles = ["Ground truth", "Model"]
    for k, (x0, xt) in enumerate(zip(x0s, xts)):
        # xt = xt[:x0.shape[0]]
        for i, xs in enumerate([x0, xt]):
            ax = fig.add_subplot(121+i, projection='mollweide') 

            display_rotations = xs @ canonical_rotation
            cmap = plt.cm.hsv
            scatterpoint_scaling = 1
            eulers_queries =_SpecialOrthogonal3Vectors().tait_bryan_angles_from_matrix(display_rotations, extrinsic_or_intrinsic="intrinsic", order="zyx")
            xyz = display_rotations[:, :, 0]
            tilt_angles = eulers_queries[:, 0]

            longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
            latitudes = np.arcsin(xyz[:, 2])

            which_to_display = np.full(longitudes.shape, True)

            # Display the distribution
            ax.scatter(
                longitudes[which_to_display],
                latitudes[which_to_display],
                s=scatterpoint_scaling,
                c=cmap(0.5 + tilt_angles[which_to_display] / 2. / np.pi))

            if dataset is not None and hasattr(dataset, 'mean'):
                # The visualization is more comprehensible if the GT
                # rotation markers are behind the output with white filling the interior.
                display_rotations_gt = dataset.mean @ canonical_rotation
            else:
                display_rotations_gt = x0 @ canonical_rotation

            for rotation in display_rotations_gt:
                _show_single_marker(ax, rotation, '*', facecolors=True)
            # Cover up the centers with white markers
            # for rotation in display_rotations_gt:
            #     _show_single_marker(ax, rotation, 'o', edgecolors=False,
            #                         facecolors='#ffffff')
            
            ax.grid()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(titles[i])

        if show_color_wheel:
            # Add a color wheel showing the tilt angle to color conversion.
            ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
            theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
            radii = np.linspace(0.4, 0.5, 2)
            _, theta_grid = np.meshgrid(radii, theta)
            colormap_val = 0.5 + theta_grid / np.pi / 2.
            ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
            ax.set_yticklabels([])
            ax.set_xticklabels([r'90$\degree$', None,
                                r'180$\degree$', None,
                                r'270$\degree$', None,
                                r'0$\degree$'], fontsize=14)
            ax.spines['polar'].set_visible(False)
            plt.text(0.5, 0.5, 'Tilt', fontsize=14,
                    horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)

    if close:
        plt.close(fig)
    return fig


def plot_so3d(x0s, xts, size, dataset=None, close=True, **kwargs):
    from packaging.version import parse as parse_version
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib import gridspec

    # Define a custom _axes3D function based on the matplotlib version.
    # The auto_add_to_figure keyword is new for matplotlib>=3.4.
    if parse_version(matplotlib.__version__) >= parse_version('3.4'):
        def _axes3D(fig, *args, **kwargs):
            ax = Axes3D(fig, *args, auto_add_to_figure=False, **kwargs)
            return fig.add_axes(ax)
    else:
        def _axes3D(*args, **kwargs):
            return Axes3D(*args, **kwargs)

    class Arrow3D(FancyArrowPatch):
        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def draw(self, renderer):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)

            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

        def do_3d_projection(self, renderer=None):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return np.min(zs)

    nrow = 2
    ncol = 3
    fig = plt.figure(figsize=(1.5*size, size))
    subfigs = fig.subfigures(nrows=nrow, ncols=1)
    titles = ["Ground truth", "Model"]

    coords_axes = np.eye(3)
    coords_axes_labels = ['x', 'y', 'z']

    for k, (x0, xt) in enumerate(zip(x0s, xts)):
        # xt = xt[:x0.shape[0]]
        for i, xs in enumerate([x0, xt]):
            gs = gridspec.GridSpec(1, ncol, wspace=0.0, hspace=0.0) 
            subfig = subfigs[i]
            subfig.suptitle(titles[i], fontsize=20)

            for j in range(3):
                ax = subfig.add_subplot(gs[0, j], projection='3d')

                # back half of sphere
                u = np.linspace(0, np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color='linen', linewidth=0, alpha=0.2, rstride=2, cstride=2)

                # wireframe
                # ax.plot_wireframe(x, y, z, rstride=20, cstride=10, color='gray', alpha=0.2)
                # equator
                # ax.plot(1.0 * np.cos(u), 1.0 * np.sin(u), zs=0, zdir='z', lw=1, color='gray')
                # ax.plot(1.0 * np.cos(u), 1.0 * np.sin(u), zs=0, zdir='x', lw=1, color='gray')
                
                # front half of sphere
                u = np.linspace(-np.pi, 0, 100)
                v = np.linspace(0, np.pi, 100)
                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color='linen', linewidth=0, alpha=0.2, rstride=2, cstride=2)

                # wireframe
                # ax.plot_wireframe(x, y, z, rstride=20, cstride=10, color='gray', alpha=0.2)
                # equator
                # ax.plot(1.0 * np.cos(u), 1.0 * np.sin(u), zs=0, zdir='z', lw=1, color='gray')
                # ax.plot(1.0 * np.cos(u), 1.0 * np.sin(u), zs=0, zdir='x', lw=1, color='gray')

                # # plot circular curves over the surface
                # theta = np.linspace(0, 2 * np.pi, 100)
                # z = np.zeros(100)
                # x = np.sin(theta)
                # y = np.cos(theta)

                # ax.plot(x, y, z, color='black', alpha=0.4)
                # ax.plot(z, x, y, color='black', alpha=0.4)

                ## add axis lines
                zeros = np.zeros(100)
                line = np.linspace(-1,1,100)

                ax.plot(line, zeros, zeros, color='black', alpha=0.4)
                ax.plot(zeros, line, zeros, color='black', alpha=0.4)
                ax.plot(zeros, zeros, line, color='black', alpha=0.4)
                ax.set_box_aspect((1, 1, 1))

                
                coords_axis = coords_axes[j]
                a = Arrow3D(0, 0, 0, coords_axis[0]*1.05, coords_axis[1]*1.05, coords_axis[2]*1.05,
                            mutation_scale=20,
                            lw=3,
                            arrowstyle='-|>',
                            color=f"C{j}", alpha=1)

                ax.add_artist(a)

                ax.text(coords_axis[0] * 1.25, coords_axis[1] * 1.25, coords_axis[2] * 1.25, coords_axes_labels[j], 
                        fontsize=16, color=f"C{j}", horizontalalignment='center', verticalalignment='center')
                

                plt.axis('off')
                plt.tight_layout()

                xsv = xs @ coords_axis

                if i == 0:
                    ax.scatter(xsv[:, 0], xsv[:, 1], xsv[:, 2], color=f"C{j}", s=8)
                else:
                    ax.scatter(xsv[:, 0], xsv[:, 1], xsv[:, 2], color=f"C{j}", s=1)

    if close:
        plt.close(fig)
    return fig


def plot_normal(x, dim, size=10):
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

    plt.close(fig)
    return fig


def plot(manifold, x0, xt, dataset=None, prob=None, size=10, close=True):
    if isinstance(manifold, Euclidean) and manifold.dim == 3:
        fig = plot_3d(x0, xt, size, prob=prob)
    elif isinstance(manifold, Hypersphere) and manifold.dim == 2:
        fig = plot_3d(x0, xt, size, prob=prob)
    elif isinstance(manifold, _SpecialOrthogonalMatrices) and manifold.dim == 3:
        fig1 = plot_so3(x0, xt, size, dataset=dataset, prob=prob, close=close)
        fig2 = plot_so3c(x0, xt, size, dataset=dataset, prob=prob, close=close)
        fig3 = plot_so3d(x0, xt, size, dataset=dataset, prob=prob, close=close)
        return [fig1, fig2, fig3]
    elif (
        isinstance(manifold, ProductSameManifold)
        and isinstance(manifold.manifold, Hypersphere)
        and manifold.manifold.dim == 1
        and manifold.dim == 1
    ) or (isinstance(manifold, Hypersphere) and manifold.dim == 1):
        fig = plot_t1(x0, xt, size, prob=prob)
    elif (
        isinstance(manifold, ProductSameManifold)
        and isinstance(manifold.manifold, Hypersphere)
        and manifold.manifold.dim == 1
    ):
        fig = plot_tn(x0, xt, size, prob=prob)
    else:
        print("Only plotting over R^3, S^2, S1/T1, T2, TN and SO(3) is implemented.")
        return None
    return fig


def plot_ref(manifold, xt, size=10, close=True):
    if isinstance(manifold, Euclidean):
        fig = plot_normal(xt, manifold.dim, size)
    elif isinstance(manifold, Hypersphere) and manifold.dim == 2:
        fig = None
    elif isinstance(manifold, _SpecialOrthogonalMatrices) and manifold.dim == 3:
        fig = plot_so3_uniform(xt, size, close=close)
    else:
        print("Only plotting over R^3, S^2 and SO(3) is implemented.")
        return None
    return fig


if __name__ == "__main__":
    from riemannian_score_sde.datasets import Wrapped
    import jax
    import jax.numpy as jnp

    SO3 = _SpecialOrthogonalMatrices(3)
    # dataset = Wrapped(100, "random", 16, [500], SO3, 42, False, 'unif')
    # x0s, context = next(dataset)
    rng = jax.random.PRNGKey(1)

    # # rng, next_rng = jax.random.split(rng)
    # # x0s = SO3.random_uniform(state=next_rng, n_samples=5)
    rng, next_rng = jax.random.split(rng)
    # xts = SO3.random_uniform(state=next_rng, n_samples=5)

    # fig = plot_so3d(x0s, xts, 10, dataset=dataset)
    # fig.savefig("test.png")


    samples = SO3.random_uniform(state=next_rng)
    theta_y = -jnp.arcsin(samples[:, 2, 0])  # -latitudes
    sign_cos_theta_y = jnp.sign(jnp.cos(theta_y))
    print(sign_cos_theta_y)
    theta_z = jnp.arctan2(samples[:, 1, 0] * sign_cos_theta_y, samples[:, 0, 0] * sign_cos_theta_y)  # longitudes
    theta_x = jnp.arctan2(samples[:, 2, 1] * sign_cos_theta_y, samples[:, 2, 2] * sign_cos_theta_y)
    # theta_z = jnp.arctan(samples[:, 0, 0] / samples[:, 1, 0])
    # theta_x = jnp.arctan(samples[:, 2, 2] / samples[:, 2, 1])
    print(jnp.stack((theta_x, theta_y, theta_z), axis=-1))

    print(_SpecialOrthogonal3Vectors().tait_bryan_angles_from_matrix(samples, extrinsic_or_intrinsic="intrinsic", order="zyx"))
