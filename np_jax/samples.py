import jax_models as jxm
import numpy as np
import jax
import jax.numpy as jnp
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from sacred import Experiment
from approximations.euclidean import run_euclidean
from approximations.bergamin import run_bergamin
from approximations.monge import run_monge
from approximations.fisher import run_fisher
from utils import get_plot_configs

ex = Experiment("main")

# ChatGPT
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


def get_samples(
    dim,
    neg_hessian_fn,
    fisher_metric_fn,
    bergamin_christoffel_fn,
    name,
    hat_theta,
    num_samples=2000,
    euclidean=True,
    bergamin=True,
    monge=True,
    fisher=True,
    empirical_fisher=False,
    fisher_christoffel_fn=None,
    empirical_fisher_christoffel_fn=None,
    ground_truth_samples=None,
    save_figures=False,
    save_samples=False,
    use_diffrax=False,
    run=None,
    xlim=None,
    ylim=None,
    true_dist_levels=None,
    true_dist_colors=None,
    contours=None,
    figsize=(10, 10),
    logger=None,
    run_hessian_precision=True,
    run_fisher_precision=False,
):
    if save_figures:
        assert dim == 2
    results = dict()

    if euclidean:
        if run_hessian_precision:
            run_euclidean(
                precision_fn=neg_hessian_fn,
                hat_theta=hat_theta,
                num_samples=num_samples,
                ground_truth_samples=ground_truth_samples,
                results=results,
                save_figures=save_figures,
                save_samples=save_samples,
                run=run,
                name=name,
                fisher_precision=False,
                xlim=xlim,
                ylim=ylim,
                true_dist_levels=true_dist_levels,
                true_dist_colors=true_dist_colors,
                contours=contours,
                figsize=figsize,
                logger=logger,
            )

        elif run_fisher_precision:
            run_euclidean(
                precision_fn=fisher_metric_fn,
                hat_theta=hat_theta,
                num_samples=num_samples,
                ground_truth_samples=ground_truth_samples,
                results=results,
                save_figures=save_figures,
                save_samples=save_samples,
                run=run,
                name=name,
                fisher_precision=True,
                xlim=xlim,
                ylim=ylim,
                true_dist_levels=true_dist_levels,
                true_dist_colors=true_dist_colors,
                contours=contours,
                figsize=figsize,
                logger=logger,
            )

    if bergamin:
        if run_hessian_precision:
            run_bergamin(
                precision_fn=neg_hessian_fn,
                hat_theta=hat_theta,
                num_samples=num_samples,
                use_diffrax=use_diffrax,
                dim=dim,
                bergamin_christoffel_fn=bergamin_christoffel_fn,
                save_figures=save_figures,
                save_samples=save_samples,
                ground_truth_samples=ground_truth_samples,
                results=results,
                run=run,
                name=name,
                fisher_precision=False,
                xlim=xlim,
                ylim=ylim,
                true_dist_levels=true_dist_levels,
                true_dist_colors=true_dist_colors,
                contours=contours,
                figsize=figsize,
                logger=logger,
            )

        elif run_fisher_precision:
            run_bergamin(
                precision_fn=fisher_metric_fn,
                hat_theta=hat_theta,
                num_samples=num_samples,
                use_diffrax=use_diffrax,
                dim=dim,
                bergamin_christoffel_fn=bergamin_christoffel_fn,
                save_figures=save_figures,
                save_samples=save_samples,
                ground_truth_samples=ground_truth_samples,
                results=results,
                run=run,
                name=name,
                fisher_precision=True,
                xlim=xlim,
                ylim=ylim,
                true_dist_levels=true_dist_levels,
                true_dist_colors=true_dist_colors,
                contours=contours,
                figsize=figsize,
                logger=logger,
            )

    if monge:
        if run_hessian_precision:
            run_monge(
                precision_fn=neg_hessian_fn,
                hat_theta=hat_theta,
                num_samples=num_samples,
                use_diffrax=use_diffrax,
                dim=dim,
                bergamin_christoffel_fn=bergamin_christoffel_fn,
                ground_truth_samples=ground_truth_samples,
                results=results,
                save_figures=save_figures,
                save_samples=save_samples,
                run=run,
                name=name,
                fisher_precision=False,
                xlim=xlim,
                ylim=ylim,
                true_dist_levels=true_dist_levels,
                true_dist_colors=true_dist_colors,
                contours=contours,
                figsize=figsize,
                logger=logger,
            )

        elif run_fisher_precision:
            run_monge(
                precision_fn=fisher_metric_fn,
                hat_theta=hat_theta,
                num_samples=num_samples,
                use_diffrax=use_diffrax,
                dim=dim,
                bergamin_christoffel_fn=bergamin_christoffel_fn,
                ground_truth_samples=ground_truth_samples,
                results=results,
                save_figures=save_figures,
                save_samples=save_samples,
                run=run,
                name=name,
                fisher_precision=True,
                xlim=xlim,
                ylim=ylim,
                true_dist_levels=true_dist_levels,
                true_dist_colors=true_dist_colors,
                contours=contours,
                figsize=figsize,
                logger=logger,
            )

    if fisher:
        if run_hessian_precision:
            run_fisher(
                fisher_christoffel_fn=fisher_christoffel_fn,
                precision_fn=neg_hessian_fn,
                hat_theta=hat_theta,
                num_samples=num_samples,
                use_diffrax=use_diffrax,
                dim=dim,
                ground_truth_samples=ground_truth_samples,
                results=results,
                save_figures=save_figures,
                save_samples=save_samples,
                run=run,
                name=name,
                fisher_precision=False,
                xlim=xlim,
                ylim=ylim,
                true_dist_levels=true_dist_levels,
                true_dist_colors=true_dist_colors,
                contours=contours,
                figsize=figsize,
                logger=logger,
                is_empirical=False,
            )

        elif run_fisher_precision:
            run_fisher(
                fisher_christoffel_fn=fisher_christoffel_fn,
                precision_fn=fisher_metric_fn,
                hat_theta=hat_theta,
                num_samples=num_samples,
                use_diffrax=use_diffrax,
                dim=dim,
                ground_truth_samples=ground_truth_samples,
                results=results,
                save_figures=save_figures,
                save_samples=save_samples,
                run=run,
                name=name,
                fisher_precision=True,
                xlim=xlim,
                ylim=ylim,
                true_dist_levels=true_dist_levels,
                true_dist_colors=true_dist_colors,
                contours=contours,
                figsize=figsize,
                logger=logger,
                is_empirical=False,
            )

    if empirical_fisher:
        if run_hessian_precision:
            run_fisher(
                fisher_christoffel_fn=empirical_fisher_christoffel_fn,
                precision_fn=neg_hessian_fn,
                hat_theta=hat_theta,
                num_samples=num_samples,
                use_diffrax=use_diffrax,
                dim=dim,
                ground_truth_samples=ground_truth_samples,
                results=results,
                save_figures=save_figures,
                save_samples=save_samples,
                run=run,
                name=name,
                fisher_precision=False,
                xlim=xlim,
                ylim=ylim,
                true_dist_levels=true_dist_levels,
                true_dist_colors=true_dist_colors,
                contours=contours,
                figsize=figsize,
                logger=logger,
                is_empirical=True,
            )

        elif run_fisher_precision:
            run_fisher(
                fisher_christoffel_fn=empirical_fisher_christoffel_fn,
                precision_fn=fisher_metric_fn,
                hat_theta=hat_theta,
                num_samples=num_samples,
                use_diffrax=use_diffrax,
                dim=dim,
                ground_truth_samples=ground_truth_samples,
                results=results,
                save_figures=save_figures,
                save_samples=save_samples,
                run=run,
                name=name,
                fisher_precision=True,
                xlim=xlim,
                ylim=ylim,
                true_dist_levels=true_dist_levels,
                true_dist_colors=true_dist_colors,
                contours=contours,
                figsize=figsize,
                logger=logger,
                is_empirical=True,
            )

    return results


def get_funcs(model, hat_theta, standardized, logger):
    xlim = None
    ylim = None
    true_dist_levels = None
    true_dist_colors = None
    contours = None
    figsize = None
    neg_hessian_fn = None
    bergamin_christoffel_fn = None

    # dummy function
    def empirical_fisher_christoffel_fn(theta):
        pass

    if model == "gaussian":
        dim, logp_fn, fisher_metric_fn, fisher_christoffel_fn = jxm.gaussian()

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
    elif model == "squiggle_easy":
        (
            dim,
            _,
            _,
            logp_fn,
            _,
            fisher_metric_fn,
            fisher_christoffel_fn,
        ) = jxm.squiggle("easy")
        # they coincide
        neg_hessian_fn = fisher_metric_fn

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
    elif model == "squiggle_difficult":
        (
            dim,
            _,
            _,
            logp_fn,
            _,
            fisher_metric_fn,
            fisher_christoffel_fn,
        ) = jxm.squiggle("difficult")
        # they coincide
        neg_hessian_fn = fisher_metric_fn

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
    elif model == "funnel":
        dim, logp_fn, _, fisher_metric_fn, fisher_christoffel_fn = jxm.funnel()

        (
            xlim,
            ylim,
            true_dist_levels,
            true_dist_colors,
            contours,
            figsize,
        ) = get_plot_configs(model, logp_fn, logger)
        # they coincide
        neg_hessian_fn = fisher_metric_fn

        map_file = os.path.join(current_directory, f"map_estimates/{model}.npy")
        samples_file = os.path.join(
            current_directory, f"ground_truth_samples/{model}.npy"
        )
    elif model == "banana":
        (
            dim,
            logp_fn,
            fisher_metric_fn,
            fisher_christoffel_fn,
            empirical_fisher_christoffel_fn,
        ) = jxm.banana()

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
    elif model == "banana_hausdorff":
        (
            dim,
            logp_fn,
            fisher_metric_fn,
            fisher_christoffel_fn,
            empirical_fisher_christoffel_fn,
        ) = jxm.banana()

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
            current_directory, "ground_truth_samples/banana.npy"
        )
    elif model in [
        "lr_ripley",
        "lr_pima",
        "lr_heart",
        "lr_australian",
        "lr_german",
    ]:
        if model == "lr_ripley":
            model_fn = jxm.lr_ripley
        elif model == "lr_pima":
            model_fn = jxm.lr_pima
        elif model == "lr_heart":
            model_fn = jxm.lr_heart
        elif model == "lr_australian":
            model_fn = jxm.lr_australian
        elif model == "lr_german":
            model_fn = jxm.lr_german

        (
            dim,
            logp_fn,
            fisher_metric_fn,
            fisher_christoffel_fn,
            empirical_fisher_christoffel_fn,
            _,
            bergamin_christoffel_fn,
        ) = model_fn(standardized=standardized)
        # they coincide
        neg_hessian_fn = fisher_metric_fn

        map_file = os.path.join(
            current_directory, f"map_estimates/{model}_{standardized}.npy"
        )
        samples_file = os.path.join(
            current_directory,
            f"ground_truth_samples/{model}_{standardized}.npy",
        )
    else:
        raise Exception

    if hat_theta is None:
        hat_theta = np.load(map_file)
    ground_truth_samples = np.load(samples_file)

    if neg_hessian_fn is None:
        neg_hessian_fn = lambda theta: -jax.hessian(logp_fn)(theta)

    if bergamin_christoffel_fn is None:
        jax_grad_and_hvp_fn = lambda theta, v: jax.jvp(jax.grad(logp_fn), [theta], [v])

        def bergamin_christoffel_fn(theta, v):
            theta_grad, theta_hvp_v = jax_grad_and_hvp_fn(theta, v)
            norm_theta_grad_2 = jnp.dot(theta_grad, theta_grad)

            W_2 = 1.0 + norm_theta_grad_2
            mho = jnp.dot(v, theta_hvp_v) / W_2
            return mho * theta_grad

    return (
        dim,
        hat_theta,
        jax.jit(neg_hessian_fn),
        ground_truth_samples,
        jax.jit(bergamin_christoffel_fn),
        jax.jit(fisher_metric_fn),
        jax.jit(fisher_christoffel_fn),
        jax.jit(empirical_fisher_christoffel_fn),
        xlim,
        ylim,
        true_dist_levels,
        true_dist_colors,
        contours,
        figsize,
    )


@ex.config
def my_config():
    model = None
    num_samples = 10000
    euclidean = True
    bergamin = False
    monge = False
    fisher = True
    empirical_fisher = False
    save_figures = False
    save_samples = False
    use_diffrax = True
    standardized = False
    run_hessian_precision = True
    run_fisher_precision = False
    calc_metric = True


@ex.automain
def my_main(
    model,
    num_samples,
    euclidean,
    bergamin,
    monge,
    fisher,
    empirical_fisher,
    save_figures,
    save_samples,
    use_diffrax,
    standardized,
    run_hessian_precision,
    run_fisher_precision,
    calc_metric,
    _run,
    _log,
):
    ground_truth_samples = None
    fisher_christoffel_fn = None

    hat_theta = None

    (
        dim,
        hat_theta,
        neg_hessian_fn,
        ground_truth_samples,
        bergamin_christoffel_fn,
        fisher_metric_fn,
        fisher_christoffel_fn,
        empirical_fisher_christoffel_fn,
        xlim,
        ylim,
        true_dist_levels,
        true_dist_colors,
        contours,
        figsize,
    ) = get_funcs(model, hat_theta, standardized, _log)
    if not calc_metric:
        ground_truth_samples = None
    if empirical_fisher:
        assert empirical_fisher_christoffel_fn is not None

    results = get_samples(
        dim=dim,
        neg_hessian_fn=neg_hessian_fn,
        bergamin_christoffel_fn=bergamin_christoffel_fn,
        name=model,
        hat_theta=hat_theta,
        euclidean=euclidean,
        bergamin=bergamin,
        monge=monge,
        fisher=fisher,
        empirical_fisher=empirical_fisher,
        fisher_metric_fn=fisher_metric_fn,
        fisher_christoffel_fn=fisher_christoffel_fn,
        empirical_fisher_christoffel_fn=empirical_fisher_christoffel_fn,
        num_samples=num_samples,
        ground_truth_samples=ground_truth_samples,
        save_figures=save_figures,
        save_samples=save_samples,
        use_diffrax=use_diffrax,
        run=_run,
        xlim=xlim,
        ylim=ylim,
        true_dist_levels=true_dist_levels,
        true_dist_colors=true_dist_colors,
        contours=contours,
        figsize=figsize,
        logger=_log,
        run_hessian_precision=run_hessian_precision,
        run_fisher_precision=run_fisher_precision,
    )

    return results
