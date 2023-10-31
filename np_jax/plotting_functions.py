import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from utils import get_show_function
from scipy import integrate


def sub_sample(samples, max_samples=1000):
    n_rows, _ = samples.shape
    if n_rows <= max_samples:
        return samples
    if n_rows > max_samples:
        id_samples = np.random.choice(n_rows, size=max_samples, replace=False)
        return samples[id_samples]


def sub_samples(samples1, samples2, max_samples=1000):
    n_rows, _ = samples1.shape
    assert samples1.shape == samples2.shape
    if n_rows <= max_samples:
        return samples1, samples2
    if n_rows > max_samples:
        id_samples = np.random.choice(n_rows, size=max_samples, replace=False)
        return samples1[id_samples], samples2[id_samples]


def get_plot(
    name,
    samples,
    contours,
    xlim,
    ylim,
    hat_theta=None,
    figsize=(15, 10),
    file_name="figs/samples.png",
    xaxes=[r"$\theta_1$", r"$\theta_2$"],
    true_dist_levels=None,
    true_dist_colors=None,
    logger=None,
):
    if name == "gaussian" or name.startswith("squiggle") or name == "funnel":
        if name == "gaussian":
            density_x = lambda x: sps.norm.pdf(x, loc=0.0, scale=1.0)
            density_y = lambda y: sps.norm.pdf(y, loc=0.0, scale=1.0)
        elif name == "squiggle_easy":
            Sigma = np.array([[5.0, 0.0], [0.0, 0.05]])
            a = 1.5

            density_x = lambda x: sps.norm.pdf(x, loc=0.0, scale=np.sqrt(5.0))

            def logp_fn(theta):
                return sps.multivariate_normal.logpdf(
                    np.zeros(2),
                    np.array([theta[0], theta[1] + np.sin(a * theta[0])]),
                    Sigma,
                )

            def marginal_integrator(y):
                I, _ = integrate.quad(lambda x: np.exp(logp_fn([x, y])), -15.0, 15.0)
                return I

            density_y = np.vectorize(marginal_integrator)
        elif name == "squiggle_difficult":
            Sigma = np.array([[10.0, 0.0], [0.0, 0.001]])
            a = 1.5

            density_x = lambda x: sps.norm.pdf(x, loc=0.0, scale=np.sqrt(10.0))

            def logp_fn(theta):
                return sps.multivariate_normal.logpdf(
                    np.zeros(2),
                    np.array([theta[0], theta[1] + np.sin(a * theta[0])]),
                    Sigma,
                )

            def marginal_integrator(y):
                I, _ = integrate.quad(
                    lambda x: np.exp(logp_fn([x, y])), -8.0, 8.0, limit=100
                )
                return I

            density_y = np.vectorize(marginal_integrator)
        elif name == "funnel":

            def logp_fn(theta):
                return sps.norm.logpdf(theta[1], loc=0.0, scale=3.0) + sps.norm.logpdf(
                    theta[0], loc=0.0, scale=np.exp(0.5 * theta[1])
                )

            def marginal_integrator(x):
                I, _ = integrate.quad(lambda y: np.exp(logp_fn([x, y])), -20.0, 20.0)
                return I

            density_x = np.vectorize(marginal_integrator)

            density_y = lambda y: sps.norm.pdf(y, loc=0.0, scale=3.0)

        plot_marginal(
            samples,
            contours,
            xlim,
            ylim,
            density_x,
            density_y,
            hat_theta=hat_theta,
            figsize=figsize,
            file_name=file_name,
            xaxes=xaxes,
            true_dist_levels=true_dist_levels,
            true_dist_colors=true_dist_colors,
            logger=logger,
        )
    else:
        plot_samples(
            samples,
            contours,
            xlim,
            ylim,
            hat_theta=hat_theta,
            figsize=figsize,
            file_name=file_name,
            xaxes=xaxes,
            true_dist_levels=true_dist_levels,
            true_dist_colors=true_dist_colors,
            logger=logger,
        )


def plot_samples(
    samples,
    contours,
    xlim,
    ylim,
    hat_theta=None,
    figsize=(10, 10),
    file_name="figs/samples.png",
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

    plt.xlabel(xaxes[0])
    plt.ylabel(xaxes[1])
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

    sub_samples = sub_sample(samples)
    plt.scatter(
        sub_samples[:, 0],
        sub_samples[:, 1],
        alpha=0.5,
        s=20,
        marker="o",
        zorder=2,
        c="dodgerblue",
    )

    plt.savefig(file_name, dpi=200, bbox_inches="tight")
    show("Plot saved")


def plot_marginal(
    samples,
    contours,
    xlim,
    ylim,
    density_x,
    density_y,
    hat_theta=None,
    figsize=(15, 10),
    file_name="figs/samples.png",
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
    plt.figure(figsize=figsize)

    sub_samples = sub_sample(samples, max_samples=1000)
    # Plot everything together

    fig, axs = plt.subplots(
        2,
        2,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1, 3], "width_ratios": [3, 1]},
    )

    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_yticklabels([])

    axs[0, 0].set_xlim(xlim[0], xlim[1])
    indexes = (samples[:, 0] >= xlim[0]) & (samples[:, 0] <= xlim[1])
    axs[0, 0].hist(
        samples[indexes][:, 0],
        bins=50,
        density=True,
        edgecolor="lightsteelblue",
        facecolor="dodgerblue",
    )

    xs = np.arange(xlim[0], xlim[1], 0.01)
    axs[0, 0].plot(xs, density_x(xs), c="black", linewidth=1.5)

    # https://stackoverflow.com/a/10035974
    axs[0, 1].axis("off")

    axs[1, 0].contour(
        X,
        Y,
        Z,
        levels=true_dist_levels,
        colors=true_dist_colors,
        linestyles="dashed",
        linewidths=1.5,
    )
    axs[1, 0].set_xlabel(xaxes[0])
    axs[1, 0].set_ylabel(xaxes[1])
    axs[1, 0].set_xlim(xlim[0], xlim[1])
    axs[1, 0].set_ylim(ylim[0], ylim[1])
    if hat_theta is not None:
        axs[1, 0].plot(
            hat_theta[0],
            hat_theta[1],
            zorder=5,
            marker="o",
            c="red",
            markersize=15,
        )
    axs[1, 0].scatter(
        sub_samples[:, 0],
        sub_samples[:, 1],
        alpha=0.5,
        s=20,
        marker="o",
        zorder=2,
        color="dodgerblue",
    )

    axs[1, 1].set_xticklabels([])
    axs[1, 1].set_yticklabels([])

    axs[1, 1].set_ylim(ylim[0], ylim[1])
    indexes = (samples[:, 1] >= ylim[0]) & (samples[:, 1] <= ylim[1])
    axs[1, 1].hist(
        samples[indexes][:, 1],
        bins=50,
        density=True,
        edgecolor="lightsteelblue",
        facecolor="dodgerblue",
        orientation="horizontal",
    )

    ys = np.arange(ylim[0], ylim[1], 0.01)
    axs[1, 1].plot(density_y(ys), ys, c="black", linewidth=1.5)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.savefig(file_name, dpi=200, bbox_inches="tight")

    show("Plot saved")


def plot_distribution(
    contours,
    xlim,
    ylim,
    hat_theta=None,
    figsize=(10, 10),
    file_name="figs/distribution.png",
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

    plt.xlabel(xaxes[0])
    plt.ylabel(xaxes[1])
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    # ChatGPT
    plt.xticks([])
    plt.yticks([])

    plt.savefig(file_name, dpi=200, bbox_inches="tight")
    show("Plot saved")
