import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.scipy.stats as jsps
import os
import numpy as np
import scipy.stats as sps

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from sacred import Experiment
from utils import get_1d_wasserstein, get_show_function, sneaky_artifact
from get_geodesic_quantities import jax_christoffel_geodesic

ex = Experiment("main")

# ChatGPT
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

dims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num_samples = 10000

samples_file = os.path.join(current_directory, "ground_truth_samples/1d_gaussian.npy")
ground_truth_samples = np.load(samples_file)


@ex.config
def my_config():
    euclidean = False
    bergamin = True
    repeats = 5


@ex.automain
def my_main(euclidean, bergamin, repeats, _run, _log):
    results = dict()
    show = get_show_function(_log)
    for dim in dims:
        show(dim)
        results[dim] = dict()

        logp_fn = lambda theta: jsps.multivariate_normal.logpdf(
            theta, jnp.zeros(dim), jnp.eye(dim)
        )

        precision_fn = lambda theta: -jax.hessian(logp_fn)(theta)
        hat_theta = jnp.zeros(dim)

        precision = np.asarray(precision_fn(hat_theta))
        precision = 0.5 * (precision + precision.T)
        covariance = sps.Covariance.from_precision(precision)

        if euclidean:
            results[dim]["Euclidean"] = dict()
            results[dim]["Euclidean"]["distances"] = np.zeros(repeats)

        if bergamin:
            results[dim]["Bergamin"] = dict()
            results[dim]["Bergamin"]["distances"] = np.zeros(repeats)

            jax_grad_and_hvp_fn = lambda theta, v: jax.jvp(
                jax.grad(logp_fn), [theta], [v]
            )

            def bergamin_christoffel_fn(theta, v):
                theta_grad, theta_hvp_v = jax_grad_and_hvp_fn(theta, v)
                norm_theta_grad_2 = jnp.dot(theta_grad, theta_grad)

                W_2 = 1.0 + norm_theta_grad_2
                mho = jnp.dot(v, theta_hvp_v) / W_2
                return mho * theta_grad

            bergamin_christoffel_fn = jax.jit(bergamin_christoffel_fn)

        for repeat in range(repeats):

            if euclidean:
                show("Euclidean")

                temp_samples = sps.multivariate_normal.rvs(
                    mean=np.zeros(dim), cov=covariance, size=num_samples
                )

                distance = get_1d_wasserstein(
                    ground_truth_samples, temp_samples[:, 0], _log
                )

                results[dim]["Euclidean"]["distances"][repeat] = distance

            if bergamin:
                show("Bergamin")

                temp_samples = sps.multivariate_normal.rvs(
                    mean=np.zeros(dim), cov=covariance, size=num_samples
                )

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
                else:
                    max_steps_exceeded = False

                assert not max_steps_exceeded

                samples = np.asarray(samples)
                distance = get_1d_wasserstein(ground_truth_samples, samples[:, 0], _log)

                results[dim]["Bergamin"]["distances"][repeat] = distance

        if euclidean:
            results[dim]["Euclidean"]["mean"] = np.mean(
                results[dim]["Euclidean"]["distances"]
            )
            results[dim]["Euclidean"]["std"] = np.std(
                results[dim]["Euclidean"]["distances"]
            )

        if bergamin:
            results[dim]["Bergamin"]["mean"] = np.mean(
                results[dim]["Bergamin"]["distances"]
            )
            results[dim]["Bergamin"]["std"] = np.std(
                results[dim]["Bergamin"]["distances"]
            )

    file_name = sneaky_artifact(_run, "figs", "distances.png")
    plt.rcParams["font.size"] = 25
    plt.rcParams["text.usetex"] = True
    # https://stackoverflow.com/a/74136954
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    plt.figure(figsize=(7, 5))

    if euclidean:
        euclidean_means = [results[dim]["Euclidean"]["mean"] for dim in dims]
        euclidean_lowers = [
            results[dim]["Euclidean"]["mean"] - 2.0 * results[dim]["Euclidean"]["std"]
            for dim in dims
        ]
        euclidean_uppers = [
            results[dim]["Euclidean"]["mean"] + 2.0 * results[dim]["Euclidean"]["std"]
            for dim in dims
        ]
        plt.plot(dims, euclidean_means, label="ELA", color="blue")
        plt.fill_between(
            dims, euclidean_lowers, euclidean_uppers, color="blue", alpha=0.5
        )

    if bergamin:
        bergamin_means = [results[dim]["Bergamin"]["mean"] for dim in dims]
        bergamin_lowers = [
            results[dim]["Bergamin"]["mean"] - 2.0 * results[dim]["Bergamin"]["std"]
            for dim in dims
        ]
        bergamin_uppers = [
            results[dim]["Bergamin"]["mean"] + 2.0 * results[dim]["Bergamin"]["std"]
            for dim in dims
        ]
        plt.plot(dims, bergamin_means, label="RLA-B", color="orange")
        plt.fill_between(
            dims, bergamin_lowers, bergamin_uppers, color="orange", alpha=0.5
        )

    plt.legend()
    plt.xlabel(r"$D$")
    plt.ylabel(r"$\mathcal{W}$")

    plt.savefig(file_name, dpi=200, bbox_inches="tight")

    return results
