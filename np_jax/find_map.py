import scipy.optimize as optim
import numpy as np
import jax.numpy as jnp
from utils import get_show_function


def internal_map_finder(dim, center, logp_fn, grad_fn, x0, divisor, logger):
    # Return the point that achieves the minimum negative log-posterior (maximum log-posterior) across 20 independent runs
    show = get_show_function(logger)

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

    show("best_candidate")
    # https://stackoverflow.com/a/11389998
    best_candidate = np.asarray(positions[np.argmin(np.asarray(values)).item()])
    show(repr(best_candidate))
    return best_candidate


def map_finder(dim, center, logp_fn, grad_fn, divisor, logger=None):
    if center is None:
        center = np.zeros(dim)
    np.random.seed(1)
    x0 = center + np.random.randn(dim) / divisor
    return internal_map_finder(dim, center, logp_fn, grad_fn, x0, divisor, logger)
