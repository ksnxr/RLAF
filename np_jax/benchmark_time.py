import os
import jax_models as jxm
import jax
import jax.numpy as jnp
import time
import timeout_decorator
import numpy as np
import scipy.stats as sps
from get_geodesic_quantities import np_christoffel_geodesic
from scipy.integrate import solve_bvp
from utils import get_wasserstein, get_show_function, sneaky_artifact

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from sacred import Experiment

ex = Experiment("main")

# ChatGPT
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


# a slightly optimized but mathematically equivalent version of the function in get_geodesic_quantities.py
def get_np_monge_gaussian_fun(dim, precision):
    def func(t, y):
        theta = y[:dim, :]
        v = y[dim:, :]
        neg_theta_grad = precision @ theta

        return np.concatenate(
            [
                v,
                -np.sum(v * (precision @ v), axis=0)
                / (1.0 + np.sum(np.square(neg_theta_grad), axis=0))
                * neg_theta_grad,
            ]
        )

    return func


def get_np_velocity(dim, precision, v):
    # https://acme.byu.edu/0000017c-ccfe-da17-a5fd-cdfeff540000/acmefiles-08-bvp-2021-pdf
    fun = get_np_monge_gaussian_fun(dim, precision)

    def bc(ya, yb):
        return np.concatenate([ya[:dim], yb[:dim] - v])

    t_steps = 500
    x = np.linspace(0.0, 1.0, t_steps)

    # ChatGPT
    tile_v = np.tile(v[:, np.newaxis], (1, t_steps))
    y = np.concatenate([tile_v * x.reshape((1, t_steps)), tile_v], axis=0)

    return solve_bvp(
        fun=fun,
        bc=bc,
        x=x,
        y=y,
    )


def get_funcs(model, hat_theta, standardized, logger):
    neg_hessian_fn = None
    bergamin_christoffel_fn = None

    if model == "gaussian":
        dim, logp_fn, fisher_metric_fn, fisher_christoffel_fn = jxm.gaussian()

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

        map_file = os.path.join(current_directory, f"map_estimates/{model}.npy")
        samples_file = os.path.join(
            current_directory, f"ground_truth_samples/{model}.npy"
        )
    elif model == "funnel":
        dim, logp_fn, _, fisher_metric_fn, fisher_christoffel_fn = jxm.funnel()
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
            _,
        ) = jxm.banana()

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
            _,
        ) = jxm.banana()

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
            _,
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
    )


@ex.config
def my_config():
    model = None
    num_samples = 1000
    bergamin = True
    monge = False
    fisher = True
    standardized = False
    run_hessian_precision = True
    run_fisher_precision = False
    calc_metric = False
    save_samples = False
    save_times = True
    timeout_limit = 20


@ex.automain
def my_main(
    model,
    num_samples,
    bergamin,
    monge,
    fisher,
    standardized,
    run_hessian_precision,
    run_fisher_precision,
    calc_metric,
    save_samples,
    save_times,
    timeout_limit,
    _run,
    _log,
):
    show = get_show_function(_log)
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
    ) = get_funcs(model, hat_theta, standardized, _log)
    if not calc_metric:
        ground_truth_samples = None

    if run_hessian_precision:
        precision_type = "hessian"
        precision = np.asarray(neg_hessian_fn(jnp.asarray(hat_theta)))
    elif run_fisher_precision:
        precision_type = "fisher"
        precision = np.asarray(fisher_metric_fn(jnp.asarray(hat_theta)))

    precision = 0.5 * (precision + precision.T)
    covariance = sps.Covariance.from_precision(precision)

    temp_samples = sps.multivariate_normal.rvs(
        mean=np.zeros(dim), cov=covariance, size=num_samples
    )
    results = dict()

    def benchmark(method_name, func):
        name = f"{method_name.capitalize()}_{precision_type}_precision"
        if precision_type == "hessian":
            samples_name = f"h_{model}_{method_name}_samples.npy"
            times_name = f"h_{model}_{method_name}_times.npy"
            timeout_indexes_name = f"h_{model}_{method_name}_timeout_indexes.npy"
        elif precision_type == "fisher":
            samples_name = f"f_{model}_{method_name}_samples.npy"
            times_name = f"f_{model}_{method_name}_times.npy"
            timeout_indexes_name = f"f_{model}_{method_name}_timeout_indexes.npy"
        show(name)
        results[name] = dict()
        samples = np.zeros((num_samples, dim))
        times = np.zeros(num_samples)
        timeout_indexes = []

        for num_sample in range(num_samples):
            try:
                t1 = time.time()
                final_sample = func(num_sample)
                times[num_sample] = time.time() - t1
                samples[num_sample] = final_sample["y"][:dim, -1]
            except:
                timeout_indexes.append(num_sample)
                times[num_sample] = timeout_limit
                # dummy choice
                samples[num_sample] = temp_samples[num_sample]

        results[name]["num_timeouts"] = len(timeout_indexes)

        if calc_metric:
            distance = get_wasserstein(ground_truth_samples, samples)
            results[name]["W"] = distance

        if save_samples:
            np.save(sneaky_artifact(_run, "samples", samples_name), samples)

        if save_times:
            np.save(sneaky_artifact(_run, "times", times_name), times)
            np.save(
                sneaky_artifact(_run, "times", timeout_indexes_name),
                np.asarray(timeout_indexes),
            )

        mean_time = np.mean(times)
        std_time = np.std(times)
        show(f"mean: {mean_time}, std: {std_time}")
        show("")

    if bergamin:
        np_bergamin_christoffel_fn = lambda theta, v: np.asarray(
            bergamin_christoffel_fn(jnp.asarray(theta), jnp.asarray(v))
        )

        @timeout_decorator.timeout(timeout_limit)
        def func(num_sample):
            solution = np_christoffel_geodesic(
                dim=dim,
                christoffel_fn=np_bergamin_christoffel_fn,
                theta=hat_theta,
                v=temp_samples[num_sample],
            )
            return solution

        benchmark("bergamin", func)

    if monge:
        np_bergamin_christoffel_fn = lambda theta, v: np.asarray(
            bergamin_christoffel_fn(jnp.asarray(theta), jnp.asarray(v))
        )

        @timeout_decorator.timeout(timeout_limit)
        def func(num_sample):
            velocity = get_np_velocity(dim, precision, temp_samples[num_sample])["y"][
                dim:, 0
            ]
            solution = np_christoffel_geodesic(
                dim=dim,
                christoffel_fn=np_bergamin_christoffel_fn,
                theta=hat_theta,
                v=velocity,
            )
            return solution

        benchmark("monge", func)

    if fisher:
        np_fisher_christoffel_fn = lambda theta, v: np.asarray(
            fisher_christoffel_fn(jnp.asarray(theta), jnp.asarray(v))
        )

        @timeout_decorator.timeout(timeout_limit)
        def func(num_sample):
            solution = np_christoffel_geodesic(
                dim=dim,
                christoffel_fn=np_fisher_christoffel_fn,
                theta=hat_theta,
                v=temp_samples[num_sample],
            )
            return solution

        benchmark("fisher", func)

    return results
