import numpy as np
import jax.numpy as jnp
import scipy.stats as sps
from utils import get_wasserstein, sneaky_artifact, get_show_function
from plotting_functions import get_plot


def run_euclidean(
    precision_fn,
    hat_theta,
    num_samples,
    ground_truth_samples,
    results,
    save_figures,
    save_samples,
    run,
    name,
    fisher_precision,
    xlim,
    ylim,
    true_dist_levels,
    true_dist_colors,
    contours,
    figsize,
    logger,
):
    show = get_show_function(logger)
    try:
        show("Euclidean")
        if fisher_precision:
            show("Fisher precision")
            if save_figures:
                fig_name = f"f_{name}_euclidean.png"
            if save_samples:
                samples_name = f"f_{name}_euclidean_samples.npy"
        else:
            show("Hessian precision")
            if save_figures:
                fig_name = f"h_{name}_euclidean.png"
            if save_samples:
                samples_name = f"h_{name}_euclidean_samples.npy"

        # Improved with help from ChatGPT
        precision = np.asarray(precision_fn(jnp.asarray(hat_theta)))
        precision = 0.5 * (precision + precision.T)
        covariance = sps.Covariance.from_precision(precision)

        samples = sps.multivariate_normal.rvs(
            mean=hat_theta, cov=covariance, size=num_samples
        )

        if ground_truth_samples is not None:
            distance = get_wasserstein(ground_truth_samples, samples, logger)
            if fisher_precision:
                results["Euclidean_fisher_precision"] = {"W": distance}
            else:
                results["Euclidean_hessian_precision"] = {"W": distance}

        if save_figures:
            get_plot(
                name,
                samples,
                contours,
                xlim=xlim,
                ylim=ylim,
                hat_theta=hat_theta,
                figsize=figsize,
                file_name=sneaky_artifact(run, "figs", fig_name),
                true_dist_levels=true_dist_levels,
                true_dist_colors=true_dist_colors,
                logger=logger,
            )

        if save_samples:
            np.save(
                sneaky_artifact(run, "samples", samples_name),
                samples,
            )
        show("")
    # https://stackoverflow.com/a/61226203
    except Exception as exception:
        show(exception)
        show("The algorithm breaks down.")
