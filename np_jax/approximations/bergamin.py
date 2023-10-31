import numpy as np
import time
import jax
import jax.numpy as jnp
import scipy.stats as sps
from joblib import Parallel, delayed
from utils import get_wasserstein, sneaky_artifact, get_show_function
from get_geodesic_quantities import (
    jax_christoffel_geodesic,
    np_christoffel_geodesic,
    jax_christoffel_geodesic_const_stepsize,
)
from plotting_functions import get_plot
from utils import get_show_function


def run_bergamin(
    precision_fn,
    hat_theta,
    num_samples,
    use_diffrax,
    dim,
    bergamin_christoffel_fn,
    save_figures,
    save_samples,
    ground_truth_samples,
    results,
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
    max_steps_exceeded = False
    try:
        show("Bergamin")
        if fisher_precision:
            show("Fisher precision")
            if save_figures:
                fig_name = f"f_{name}_bergamin.png"
            if save_samples:
                samples_name = f"f_{name}_bergamin_samples.npy"
                num_evals_name = f"f_{name}_bergamin_num_evals.npy"
        else:
            show("Hessian precision")
            if save_figures:
                fig_name = f"h_{name}_bergamin.png"
            if save_samples:
                samples_name = f"h_{name}_bergamin_samples.npy"
                num_evals_name = f"h_{name}_bergamin_num_evals.npy"

        # Improved with help from ChatGPT
        precision = np.asarray(precision_fn(jnp.asarray(hat_theta)))
        precision = 0.5 * (precision + precision.T)
        covariance = sps.Covariance.from_precision(precision)

        temp_samples = sps.multivariate_normal.rvs(
            mean=np.zeros(dim), cov=covariance, size=num_samples
        )

        if use_diffrax:
            t1 = time.time()

            hat_theta = jnp.asarray(hat_theta)
            temp_samples = jnp.asarray(temp_samples)

            def func(temp_sample):
                solution = jax_christoffel_geodesic(
                    dim=dim,
                    christoffel_fn=bergamin_christoffel_fn,
                    theta=hat_theta,
                    v=temp_sample,
                )
                result = solution.result
                sample = solution.ys[0][:dim]
                num_eval = solution.stats["num_steps"]
                return result, sample, num_eval

            mapped_func = jax.vmap(func)
            samples_results, samples, num_evals = mapped_func(temp_samples)
            indexes = jnp.where(samples_results == 0)[0]
            assert (
                len(indexes) + len(jnp.where(samples_results == 2)[0])
            ) == num_samples
            if len(indexes) != num_samples:
                max_steps_exceeded = True

                samples = samples.at[indexes].get()

            show(f"Elapsed time: {time.time() - t1}")
            samples = np.asarray(samples)
            num_evals = np.asarray(num_evals)

        else:
            t1 = time.time()

            np_bergamin_christoffel_fn = lambda theta, v: np.asarray(
                bergamin_christoffel_fn(jnp.asarray(theta), jnp.asarray(v))
            )

            def func(num_sample):
                solution = np_christoffel_geodesic(
                    dim=dim,
                    christoffel_fn=np_bergamin_christoffel_fn,
                    theta=hat_theta,
                    v=temp_samples[num_sample],
                )
                new_theta = solution["y"][:dim, -1]
                num_eval = solution["nfev"]
                return new_theta, num_eval

            # https://stackoverflow.com/a/71977764
            samples, num_evals = zip(
                *Parallel(n_jobs=-1, timeout=99999)(
                    delayed(func)(num_sample) for num_sample in range(num_samples)
                )
            )
            show(f"Elapsed time: {time.time() - t1}")

            samples = np.stack(samples)
            num_evals = np.asarray(num_evals)

        mean_num = np.mean(num_evals)
        std_num = np.std(num_evals)
        show(f"number of evaluations: mean: {mean_num}, std: {std_num}")

        if ground_truth_samples is not None:
            distance = get_wasserstein(ground_truth_samples, samples, logger)
            if fisher_precision:
                results["Bergamin_fisher_precision"] = {
                    "W": distance,
                    "num_evals": [mean_num, std_num],
                    "max_steps_exceeded": max_steps_exceeded,
                }
            else:
                results["Bergamin_hessian_precision"] = {
                    "W": distance,
                    "num_evals": [mean_num, std_num],
                    "max_steps_exceeded": max_steps_exceeded,
                }

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
            np.save(
                sneaky_artifact(run, "num_evals", num_evals_name),
                num_evals,
            )
        show("")
    # https://stackoverflow.com/a/61226203
    except Exception as exception:
        show(exception)
        show("The algorithm breaks down.")
