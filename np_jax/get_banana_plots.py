import matplotlib.pyplot as plt
from utils import get_show_function, sneaky_artifact, get_plot_configs
from sacred import Experiment
import os
import jax_models as jxm
import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize as optim


ex = Experiment("main")

# ChatGPT
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


def plot_maps(
    euc_maps,
    hau_map,
    contours,
    xlim,
    ylim,
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

    for euc_map in euc_maps:
        plt.plot(
            euc_map[0],
            euc_map[1],
            zorder=5,
            marker="o",
            c="blue",
            markersize=15,
        )

    plt.plot(
        hau_map[0],
        hau_map[1],
        zorder=5,
        marker="o",
        c="brown",
        markersize=15,
    )

    plt.savefig(file_name, dpi=200, bbox_inches="tight")
    show("Plot saved")


@ex.automain
def my_main(_log, _run):
    show = get_show_function(_log)
    model = "banana"

    (
        dim,
        logp_fn,
        _,
        _,
        _,
    ) = jxm.banana()

    (
        xlim,
        ylim,
        true_dist_levels,
        true_dist_colors,
        contours,
        figsize,
    ) = get_plot_configs(model, logp_fn, _log)

    grad_fn = jax.jit(jax.grad(logp_fn))
    logp_fn = jax.jit(logp_fn)
    divisor = dim

    np.random.seed(1)
    center = np.zeros(dim)
    x0 = center + np.random.randn(dim) / divisor
    positions = []
    values = []
    fun = lambda x: np.asarray(-logp_fn(jnp.asarray(x)))
    jac = lambda x: np.asarray(-grad_fn(jnp.asarray(x)))
    while len(values) < 20:
        result = optim.minimize(
            fun=fun,
            x0=x0,
            method="BFGS",
            jac=jac,
            options={"maxiter": 1e6},
        )
        if result["fun"] != np.inf and not np.isnan(result["fun"]):
            show(result)
            positions.append(result["x"])
            values.append(result["fun"])

        x0 = center + np.random.randn(dim) / divisor

    min_value = values[np.argmin(np.asarray(values)).item()]
    euc_maps = []
    upper_map = False
    lower_map = False
    for index, value in enumerate(values):
        if value == min_value:
            position = positions[index]
            if position[1] > 0.0 and not upper_map:
                euc_maps.append(position)
                upper_map = True
            if position[1] < 0.0 and not lower_map:
                euc_maps.append(position)
                lower_map = True

    hau_map = np.load(
        os.path.join(current_directory, f"map_estimates/banana_hausdorff.npy")
    )

    results = dict()
    plot_maps(
        euc_maps,
        hau_map,
        contours,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        file_name=sneaky_artifact(_run, "figs", "MAPs.png"),
        true_dist_levels=true_dist_levels,
        true_dist_colors=true_dist_colors,
        logger=_log,
    )

    return results
