import jax_models as jxm
import numpy as np
import jax
import jax.numpy as jnp
import os
import scipy.stats as sps
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from get_geodesic_quantities import jax_christoffel_geodesic

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from sacred import Experiment
from utils import get_contours, sneaky_artifact, get_wasserstein, get_show_function

ex = Experiment("main")

# ChatGPT
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


def plot_metrics(
    metric_fn,
    xs,
    ys,
    multiplier,
    contours,
    xlim,
    ylim,
    figsize=(10, 10),
    file_name="figs/metrics.png",
    xaxes=[r"$\theta_1$", r"$\theta_2$"],
    true_dist_levels=None,
    true_dist_colors=None,
    logger=None,
):
    show = get_show_function(logger)
    # Plot everything together
    [X, Y, Z] = contours
    # ChatGPT
    plt.rcParams["font.size"] = 35
    plt.rcParams["text.usetex"] = True
    # https://stackoverflow.com/a/74136954
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    fig, ax = plt.subplots(figsize=figsize)
    plt.contour(
        X,
        Y,
        Z,
        levels=true_dist_levels,
        colors=true_dist_colors,
        linestyles="dashed",
        linewidths=1.5,
    )

    for x in xs:
        for y in ys:
            point = np.asarray([x, y])
            covariance_matrix = np.linalg.inv(metric_fn(point))

            # based on ChatGPT
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            # Compute the lengths of major and minor axes
            major_axis = multiplier * np.sqrt(eigenvalues[0])
            minor_axis = multiplier * np.sqrt(eigenvalues[1])
            ellipse = Ellipse(
                xy=point,
                width=major_axis,
                height=minor_axis,
                angle=np.degrees(np.arctan2(*eigenvectors[:, 0][::-1])),
                facecolor="none",
                edgecolor="green",
            )
            ax.add_patch(ellipse)

    # plt.xlabel(xaxes[0])
    # plt.ylabel(xaxes[1])
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    # ChatGPT
    plt.xticks([])
    plt.yticks([])

    plt.savefig(file_name, dpi=200, bbox_inches="tight")
    show("Plot saved")


def plot_geodesics(
    samples1,
    samples2,
    contours,
    target,
    xlim,
    ylim,
    hat_theta=None,
    figsize=(10, 10),
    file_name="figs/geodesics.png",
    xaxes=[r"$\theta_1$", r"$\theta_2$"],
    true_dist_levels=None,
    true_dist_colors=None,
    logger=None,
):
    show = get_show_function(logger)
    # Plot everything together
    [X, Y, Z] = contours
    # ChatGPT
    plt.rcParams["font.size"] = 22
    plt.rcParams["text.usetex"] = True
    # https://stackoverflow.com/a/74136954
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    plt.figure(figsize=figsize)
    plt.contour(
        X,
        Y,
        Z,
        levels=true_dist_levels,
        colors=true_dist_colors,
        linestyles="dashed",
        linewidths=1.5,
    )

    # plt.xlabel(xaxes[0])
    # plt.ylabel(xaxes[1])
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    # ChatGPT
    plt.xticks([])
    plt.yticks([])

    if hat_theta is not None:
        plt.plot(
            hat_theta[0],
            hat_theta[1],
            zorder=5,
            marker="o",
            c="red",
            markersize=15,
        )

    distances = np.linalg.norm(samples1 - target, axis=1)
    best_idx = np.argmin(distances)

    plt.plot(
        samples1[best_idx, 0],
        samples1[best_idx, 1],
        zorder=5,
        marker="o",
        c="peru",
        markersize=10,
    )
    plt.plot(
        samples2[best_idx, 0],
        samples2[best_idx, 1],
        zorder=5,
        marker="o",
        c="blue",
        markersize=10,
    )

    plt.scatter(
        samples1[:, 0],
        samples1[:, 1],
        alpha=0.5,
        s=20,
        marker="o",
        zorder=2,
        c="orange",
    )
    plt.scatter(
        samples2[:, 0],
        samples2[:, 1],
        alpha=0.5,
        s=40,
        marker="o",
        zorder=2,
        c="dodgerblue",
    )

    small_vector = samples2[best_idx] - samples1[best_idx]
    small_vector = small_vector / np.linalg.norm(small_vector) / 10.0
    plt.annotate(
        "",  # Text to display next to the arrow
        xy=(
            samples2[best_idx, 0] - 2 * small_vector[0],
            samples2[best_idx, 1] - 2 * small_vector[1],
        ),  # Coordinates of the end point of the arrow
        xytext=(
            samples1[best_idx, 0] + 2 * small_vector[0],
            samples1[best_idx, 1] + 2 * small_vector[1],
        ),  # Coordinates of the text label
        arrowprops=dict(
            color="darkolivegreen",
            linewidth=2.0,
            zorder=20,
            alpha=0.7,
            headwidth=15.0,
        ),  # Arrow style
    )

    mean_point = 0.5 * (samples1[best_idx] + samples2[best_idx])
    plt.annotate(
        r"$Exp_{\hat{\boldsymbol{\theta}}}(\boldsymbol{v})$",
        xy=(mean_point[0], mean_point[1]),
        xytext=(mean_point[0] - 1.5, mean_point[1] + 0.1),
    )

    plt.annotate(
        r"$\hat{\boldsymbol{\theta}}$",
        xy=(hat_theta[0], hat_theta[1]),
        xytext=(hat_theta[0] - 0.7, hat_theta[1] - 0.3),
    )

    plt.annotate(
        r"$\boldsymbol{v}$",
        xy=(samples1[best_idx, 0], samples1[best_idx, 1]),
        xytext=(samples1[best_idx, 0] - 0.5, samples1[best_idx, 1]),
    )

    plt.annotate(
        r"$\boldsymbol{\theta}$",
        xy=(samples2[best_idx, 0], samples2[best_idx, 1]),
        xytext=(samples2[best_idx, 0] - 0.2, samples2[best_idx, 1] - 0.6),
    )

    plt.savefig(file_name, dpi=200, bbox_inches="tight")
    show("Plot saved")


@ex.config
def my_config():
    num_samples = 300
    calc_metric = False
    seed = 1


@ex.automain
def my_main(num_samples, calc_metric, seed, _log, _run):
    (
        dim,
        logp_fn,
        fisher_metric_fn,
        fisher_christoffel_fn,
        _,
    ) = jxm.banana()

    xlim = [-2.7, 2.3]
    ylim = [-1.0, 4.0]
    true_dist_levels = [-210.0, -205.0, -200.0]
    true_dist_colors = ["black", "black", "black"]
    contours = get_contours(xlim, ylim, logp_fn, _log)
    figsize = (4, 4)

    fisher_metric_fn = jax.jit(fisher_metric_fn)
    np_fisher_metric_fn = lambda theta: np.asarray(fisher_metric_fn(jnp.asarray(theta)))

    plot_metrics(
        np_fisher_metric_fn,
        xs=np.linspace(xlim[0], xlim[1], 7),
        ys=np.linspace(ylim[0], ylim[1], 7),
        multiplier=0.5,
        contours=contours,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        file_name=sneaky_artifact(_run, "figs", "metrics.png"),
        true_dist_levels=true_dist_levels,
        true_dist_colors=true_dist_colors,
    )

    map_file = os.path.join(current_directory, f"map_estimates/banana_hausdorff.npy")
    hat_theta = np.load(map_file)

    if calc_metric:
        samples_file = os.path.join(
            current_directory, "ground_truth_samples/banana.npy"
        )
        ground_truth_samples = np.load(samples_file)

    # Euclidean Laplace Approximation
    # Improved with help from ChatGPT
    precision = fisher_metric_fn(hat_theta)
    precision = 0.5 * (precision + precision.T)
    covariance = sps.Covariance.from_precision(precision)

    np.random.seed(seed)
    orig_samples1 = sps.multivariate_normal.rvs(
        mean=np.zeros(dim), cov=covariance, size=num_samples
    )
    samples1 = orig_samples1 + hat_theta

    if calc_metric:
        get_wasserstein(ground_truth_samples, samples1, _log)

    temp_samples = jnp.asarray(orig_samples1)

    def func(temp_sample):
        solution = jax_christoffel_geodesic(
            dim=dim,
            christoffel_fn=fisher_christoffel_fn,
            theta=hat_theta,
            v=temp_sample,
        )
        ys = solution.ys[0][:dim]
        return ys

    mapped_func = jax.vmap(func)
    samples2 = mapped_func(temp_samples)
    samples2 = np.asarray(samples2)

    if calc_metric:
        get_wasserstein(ground_truth_samples, samples2, _log)

    plot_geodesics(
        samples1=samples1,
        samples2=samples2,
        contours=contours,
        target=np.array([1.2, 3.5]),
        xlim=xlim,
        ylim=ylim,
        hat_theta=hat_theta,
        figsize=figsize,
        file_name=sneaky_artifact(_run, "figs", "geodesics.png"),
        true_dist_levels=true_dist_levels,
        true_dist_colors=true_dist_colors,
    )
