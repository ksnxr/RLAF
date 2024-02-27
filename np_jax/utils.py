import ot
import numpy as np
import jax
import os
import sacred
from pathlib import Path
import jax.numpy as jnp
import scipy.stats as sps


def get_contours(xlim, ylim, logp_fn, logger=None):
    show = get_show_function(logger)
    x0 = jnp.arange(xlim[0], xlim[1], 0.01)
    x1 = jnp.arange(ylim[0], ylim[1], 0.01)
    X, Y = jnp.meshgrid(x0, x1)
    vec_logp_fn = jax.jit(jax.vmap(jax.vmap(logp_fn, in_axes=1), in_axes=1))
    Z = vec_logp_fn(jnp.stack([X, Y]))
    show(f"{jnp.min(Z)}, {jnp.max(Z)}")
    contours = [np.asarray(X), np.asarray(Y), np.asarray(Z)]
    return contours


def get_plot_configs(model, logp_fn, logger):
    true_dist_colors = ["black", "black", "black"]
    if model == "gaussian":
        xlim = [-3.0, 3.0]
        ylim = [-3.0, 3.0]
        dist = sps.multivariate_normal(mean=np.zeros(2), cov=np.eye(2))
        true_dist_levels = [
            dist.logpdf([3.0, 0.0]),
            dist.logpdf([2.0, 0.0]),
            dist.logpdf([1.0, 0.0]),
        ]
        contours = get_contours(xlim, ylim, logp_fn, logger)
        figsize = (6, 6)

    elif model == "squiggle_easy":
        xlim = [-10.0, 10.0]
        ylim = [-2.0, 2.0]
        true_dist_levels = [-5.0, -3.0, -1.5]
        contours = get_contours(xlim, ylim, logp_fn, logger)
        figsize = (6, 6)

    elif model == "squiggle_difficult":
        xlim = [-10.0, 10.0]
        ylim = [-2.0, 2.0]
        true_dist_levels = [-3.0, -1.0, 0.0]
        contours = get_contours(xlim, ylim, logp_fn, logger)
        figsize = (6, 6)

    elif model == "funnel":
        xlim = [-10.0, 10.0]
        ylim = [-10.0, 10.0]
        true_dist_levels = [-10.0, -5.0, -2.0]
        contours = get_contours(xlim, ylim, logp_fn, logger)
        figsize = (6, 6)

    elif model == "banana":
        xlim = [-5.5, 2.5]
        ylim = [-3.0, 3.0]
        true_dist_levels = [-210.0, -205.0, -204.0]
        contours = get_contours(xlim, ylim, logp_fn, logger)
        figsize = (5, 4)

    elif model == "banana_hausdorff":
        xlim = [-5.5, 2.5]
        ylim = [-3.0, 3.0]
        true_dist_levels = [-210.0, -205.0, -204.0]
        contours = get_contours(xlim, ylim, logp_fn, logger)
        figsize = (5, 4)

    return xlim, ylim, true_dist_levels, true_dist_colors, contours, figsize


def get_show_function(logger):
    if logger is not None:
        return logger.info
    else:
        return print


# Adapted from https://github.com/ratschlab/bnn_priors/blob/3597cf45a0c2496dd9e053090b3786f9fae573bb/bnn_priors/exp_utils.py#L554
def sneaky_artifact(_run, subdir, name):
    """modifed `artifact_event` from `sacred.observers.FileStorageObserver`
    Returns path to the name.
    """
    obs = _run.observers[0]
    assert isinstance(obs, sacred.observers.FileStorageObserver)
    obs.run_entry["artifacts"].append(name)
    obs.save_json(obs.run_entry, "run.json")

    subdir_path = os.path.join(Path(obs.dir), subdir)
    if not os.path.exists(subdir_path):
        # ChatGPT
        os.makedirs(subdir_path)

    return os.path.join(subdir_path, name)


def get_1d_wasserstein(samples1, samples2, logger=None):
    assert len(samples1.shape) == 1
    assert len(samples2.shape) == 1
    show = get_show_function(logger)
    distance = ot.wasserstein_1d(samples1, samples2)
    show(f"1D wasserstein distance: {distance}")
    return distance


def get_wasserstein(samples1, samples2, logger=None):
    show = get_show_function(logger)
    M = ot.dist(samples1, samples2, metric="euclidean")
    n_samples1 = samples1.shape[0]
    n_samples2 = samples2.shape[0]
    distance = ot.emd2(
        np.ones((n_samples1,)) / n_samples1,
        np.ones((n_samples2,)) / n_samples2,
        M,
        numItermax=1e10,
    )
    show(f"Wasserstein distance: {distance}")
    return distance


def get_kl(logits1, logits2, logger=None):
    show = get_show_function(logger)
    divergence = np.mean(logits1 - logits2)
    show(f"KL divergence: {divergence}")
    return divergence


def christoffel_fn(g, theta, v):
    # Adapted based on ChatGPT
    d_g = jax.jacfwd(g)(theta)

    # Compute the Christoffel symbols
    partial_1 = jnp.einsum("jli,i,j->l", d_g, v, v)
    partial_2 = jnp.einsum("ilj,i,j->l", d_g, v, v)
    partial_3 = jnp.einsum("ijl,i,j->l", d_g, v, v)
    result = jnp.linalg.solve(g(theta), 0.5 * (partial_1 + partial_2 - partial_3))

    return result
