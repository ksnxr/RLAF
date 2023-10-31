import os
import numpy as np
import jax
import jax_models as jxm
from get_ground_truth_samples import (
    get_lr_nuts_samples,
    get_banana_nuts_samples,
    get_gaussian_samples,
    get_squiggle_samples,
    get_funnel_samples,
)
from find_map import map_finder
from sacred import Experiment
from plotting_functions import plot_distribution, plot_samples
from utils import get_plot_configs

ex = Experiment("main")

# ChatGPT
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


def get_quantities(model, standardized, save_figures, show_progress, logger):
    xlim = None
    ylim = None
    true_dist_levels = None
    true_dist_colors = None
    contours = None
    figsize = None

    if not os.path.exists(os.path.join(current_directory, "figs")):
        os.mkdir(os.path.join(current_directory, "figs"))
    if not os.path.exists(os.path.join(current_directory, "figs/gt")):
        os.mkdir(os.path.join(current_directory, "figs/gt"))
    if not os.path.exists(os.path.join(current_directory, "figs/gt_samples")):
        os.mkdir(os.path.join(current_directory, "figs/gt_samples"))
    if not os.path.exists(os.path.join(current_directory, "map_estimates")):
        os.mkdir(os.path.join(current_directory, "map_estimates"))
    if not os.path.exists(os.path.join(current_directory, "ground_truth_samples")):
        os.mkdir(os.path.join(current_directory, "ground_truth_samples"))

    diagnose = ""
    if model == "gaussian":
        dim, logp_fn, _, _ = jxm.gaussian()

        (
            xlim,
            ylim,
            true_dist_levels,
            true_dist_colors,
            contours,
            figsize,
        ) = get_plot_configs(model, logp_fn, logger)

        map_file = os.path.join(current_directory, f"map_estimates/{model}.npy")
        samples_file = os.path.join(
            current_directory, f"ground_truth_samples/{model}.npy"
        )

        hat_theta = np.zeros(dim)
        np.save(map_file, hat_theta)

        diagnose = get_gaussian_samples(samples_file)

    elif model == "squiggle_easy":
        dim, a, Sigma, logp_fn, _, _, _ = jxm.squiggle("easy")

        (
            xlim,
            ylim,
            true_dist_levels,
            true_dist_colors,
            contours,
            figsize,
        ) = get_plot_configs(model, logp_fn, logger)

        map_file = os.path.join(current_directory, f"map_estimates/{model}.npy")
        samples_file = os.path.join(
            current_directory, f"ground_truth_samples/{model}.npy"
        )

        hat_theta = np.zeros(dim)
        np.save(map_file, hat_theta)
        diagnose = get_squiggle_samples(samples_file, a, Sigma)

    elif model == "squiggle_difficult":
        dim, a, Sigma, logp_fn, _, _, _ = jxm.squiggle("difficult")

        (
            xlim,
            ylim,
            true_dist_levels,
            true_dist_colors,
            contours,
            figsize,
        ) = get_plot_configs(model, logp_fn, logger)

        map_file = os.path.join(current_directory, f"map_estimates/{model}.npy")
        samples_file = os.path.join(
            current_directory, f"ground_truth_samples/{model}.npy"
        )

        hat_theta = np.zeros(dim)
        np.save(map_file, hat_theta)
        diagnose = get_squiggle_samples(samples_file, a, Sigma)

    elif model == "funnel":
        dim, logp_fn, _, _, _ = jxm.funnel()
        grad_fn = lambda x: np.asarray(jax.grad(logp_fn)(x))

        (
            xlim,
            ylim,
            true_dist_levels,
            true_dist_colors,
            contours,
            figsize,
        ) = get_plot_configs(model, logp_fn, logger)

        map_file = os.path.join(current_directory, f"map_estimates/{model}.npy")
        samples_file = os.path.join(
            current_directory, f"ground_truth_samples/{model}.npy"
        )

        hat_theta = np.zeros(dim)
        np.save(map_file, hat_theta)
        diagnose = get_funnel_samples(samples_file)

    elif model == "banana":
        dim, logp_fn, _, _, _ = jxm.banana()

        (
            xlim,
            ylim,
            true_dist_levels,
            true_dist_colors,
            contours,
            figsize,
        ) = get_plot_configs(model, logp_fn, logger)

        map_file = os.path.join(current_directory, f"map_estimates/{model}.npy")
        samples_file = os.path.join(
            current_directory, f"ground_truth_samples/{model}.npy"
        )
        divisor = dim

        hat_theta = map_finder(
            dim, None, jax.jit(logp_fn), jax.jit(jax.grad(logp_fn)), divisor, logger
        )
        np.save(map_file, hat_theta)
        diagnose = get_banana_nuts_samples(
            samples_file,
            show_progress=show_progress,
            logger=logger,
        )

    elif model == "banana_hausdorff":
        dim, _, logp_fn, _, _, _ = jxm.banana(hausdorff=True)

        (
            xlim,
            ylim,
            true_dist_levels,
            true_dist_colors,
            contours,
            figsize,
        ) = get_plot_configs(model, logp_fn, logger)

        map_file = os.path.join(current_directory, f"map_estimates/{model}.npy")
        samples_file = os.path.join(
            current_directory, f"ground_truth_samples/banana.npy"
        )
        divisor = dim

        hat_theta = map_finder(
            dim, None, jax.jit(logp_fn), jax.jit(jax.grad(logp_fn)), divisor, logger
        )
        np.save(map_file, hat_theta)
        diagnose = ""

    elif model in [
        "lr_ripley",
        "lr_pima",
        "lr_heart",
        "lr_australian",
        "lr_german",
    ]:
        if model == "lr_ripley":
            model_fn = jxm.lr_ripley
            file_name = os.path.join(current_directory, "data/ripley.npy")
        elif model == "lr_pima":
            model_fn = jxm.lr_pima
            file_name = os.path.join(current_directory, "data/pima.npy")
        elif model == "lr_heart":
            model_fn = jxm.lr_heart
            file_name = os.path.join(current_directory, "data/heart.npy")
        elif model == "lr_australian":
            model_fn = jxm.lr_australian
            file_name = os.path.join(current_directory, "data/australian.npy")
        elif model == "lr_german":
            model_fn = jxm.lr_german
            file_name = os.path.join(current_directory, "data/german.npy")
        else:
            raise Exception

        dim, logp_fn, _, _, _, grad_fn, _ = model_fn(standardized=standardized)

        map_file = os.path.join(
            current_directory, f"map_estimates/{model}_{standardized}.npy"
        )
        samples_file = os.path.join(
            current_directory,
            f"ground_truth_samples/{model}_{standardized}.npy",
        )

        divisor = dim

        hat_theta = map_finder(
            dim, None, jax.jit(logp_fn), jax.jit(grad_fn), divisor, logger
        )
        np.save(map_file, hat_theta)
        diagnose = get_lr_nuts_samples(
            file_name,
            samples_file,
            standardized,
            show_progress=show_progress,
            logger=logger,
        )

    else:
        raise Exception

    if save_figures:
        if "hausdorff" in model:
            dist_fig_name = f"{model}_gt_hausdorff.png"
        else:
            dist_fig_name = f"{model}_gt.png"
            ground_truth_samples = np.load(samples_file)
            plot_samples(
                ground_truth_samples,
                contours,
                xlim=xlim,
                ylim=ylim,
                figsize=figsize,
                file_name=os.path.join(
                    current_directory, f"figs/gt_samples/{model}_gt_samples.png"
                ),
                true_dist_levels=true_dist_levels,
                true_dist_colors=true_dist_colors,
                logger=logger,
            )

        plot_distribution(
            contours,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            file_name=os.path.join(current_directory, f"figs/gt/{dist_fig_name}"),
            true_dist_levels=true_dist_levels,
            true_dist_colors=true_dist_colors,
            logger=logger,
        )

    if diagnose == "":
        no_problems = True
    else:
        no_problems = "Processing complete, no problems detected." in diagnose

    return {
        "MAP": hat_theta,
        "diagnose": diagnose,
        "no_problems": no_problems,
    }


@ex.config
def my_config():
    model = None
    standardized = False
    save_figures = False
    show_progress = True


@ex.automain
def my_main(model, standardized, save_figures, show_progress, _log):
    assert model is not None
    return get_quantities(model, standardized, save_figures, show_progress, _log)
