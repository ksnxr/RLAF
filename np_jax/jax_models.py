import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as jsps
from sklearn.preprocessing import StandardScaler
import os
from utils import christoffel_fn

# ChatGPT
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


def gaussian():
    dim = 2

    def logp_fn(theta):
        return jsps.multivariate_normal.logpdf(theta, jnp.zeros(dim), jnp.eye(dim))

    def fisher_metric_fn(theta):
        return jnp.eye(dim)

    def fisher_christoffel_fn(theta, v):
        return jnp.zeros(dim)

    return dim, logp_fn, fisher_metric_fn, fisher_christoffel_fn


def squiggle(type_string="difficult"):
    dim = 2
    a = 1.5
    if type_string == "difficult":
        Sigma = jnp.array([[10.0, 0.0], [0.0, 0.001]])
    elif type_string == "easy":
        Sigma = jnp.array([[5.0, 0.0], [0.0, 0.05]])
    precision = jnp.linalg.inv(Sigma)

    def logp_fn(theta):
        return hausdorff_logp_fn(theta)

    def hausdorff_logp_fn(theta):
        return jsps.multivariate_normal.logpdf(
            jnp.zeros(2),
            jnp.array([theta[0], theta[1] + jnp.sin(a * theta[0])]),
            Sigma,
        )

    def fisher_metric_fn(theta):
        jacobian = jnp.array([[1.0, 0.0], [a * jnp.cos(a * theta[0]), 1.0]])
        metric = jacobian.T @ precision @ jacobian
        return 0.5 * (metric + metric.T)

    def fisher_christoffel_fn(theta, v):
        return christoffel_fn(fisher_metric_fn, theta, v)

    return (
        dim,
        a,
        Sigma,
        logp_fn,
        hausdorff_logp_fn,
        fisher_metric_fn,
        fisher_christoffel_fn,
    )


def banana(hausdorff=False):
    dim = 2
    ys = jnp.asarray(np.load(os.path.join(current_directory, "data/banana_ys.npy")))
    n = len(ys)
    sigma_theta = 2.0
    sigma_y = 2.0
    quantity_prior = 1.0 / sigma_theta**2
    quantity_00 = n / sigma_y**2
    quantity_01 = 2.0 * n / sigma_y**2
    quantity_11 = 4.0 * n / sigma_y**2

    def logp_fn(theta):
        return (
            jsps.norm.logpdf(theta[0], 0.0, sigma_theta)
            + jsps.norm.logpdf(theta[1], 0.0, sigma_theta)
            + jnp.sum(jsps.norm.logpdf(ys, theta[0] + jnp.square(theta[1]), sigma_y))
        )

    def hausdorff_logp_fn(theta):
        return logp_fn(theta) - 0.5 * jnp.linalg.slogdet(fisher_metric_fn(theta))[1]

    def fisher_metric_fn(theta):
        metric = jnp.array(
            [
                [quantity_prior + quantity_00, quantity_01 * theta[1]],
                [
                    quantity_01 * theta[1],
                    quantity_prior + quantity_11 * jnp.square(theta[1]),
                ],
            ]
        )
        return 0.5 * (metric + metric.T)

    def empirical_fisher_metric_fn(theta):
        metric = jnp.diag(jnp.array([quantity_prior, quantity_prior]))

        def func(y):
            c_grad = jax.grad(
                lambda c_theta: jsps.norm.logpdf(
                    y, c_theta[0] + jnp.square(c_theta[1]), sigma_y
                )
            )(theta)
            return jnp.outer(c_grad, c_grad)

        outers = jax.vmap(func)(ys)
        metric = metric + jnp.sum(outers, axis=0)
        return 0.5 * (metric + metric.T)

    def fisher_christoffel_fn(theta, v):
        return christoffel_fn(fisher_metric_fn, theta, v)

    def empirical_fisher_christoffel_fn(theta, v):
        return christoffel_fn(empirical_fisher_metric_fn, theta, v)

    if hausdorff:
        return (
            dim,
            logp_fn,
            hausdorff_logp_fn,
            fisher_metric_fn,
            fisher_christoffel_fn,
            empirical_fisher_christoffel_fn,
        )
    else:
        return (
            dim,
            logp_fn,
            fisher_metric_fn,
            fisher_christoffel_fn,
            empirical_fisher_christoffel_fn,
        )


def get_logistic_fn(file_name, standardized=False):
    # improved based on ChatGPT

    # prior variance
    alpha = 100.0
    sqrt_alpha = 10.0

    data = np.load(os.path.join(current_directory, file_name))

    X = data[:, :-1]

    if standardized:
        ss = StandardScaler()
        X = ss.fit_transform(X)

    new_x = np.ones((X.shape[0], 1))
    X = jnp.asarray(np.concatenate([X, new_x], axis=1))
    dim = X.shape[1]

    y = data[:, -1]
    y = jnp.asarray(y)
    # ChatGPT
    assert jnp.all((y == 0) | (y == 1))

    def logp_fn(theta):
        return jnp.sum(jsps.norm.logpdf(theta, 0.0, sqrt_alpha)) + jnp.sum(
            jsps.bernoulli.logpmf(y, jax.nn.sigmoid(jnp.dot(X, theta)))
        )

    def fisher_metric_fn(theta):
        preds = jax.nn.sigmoid(jnp.dot(X, theta))
        # Inspired by https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.svd.html
        metric = (X.T * preds * (1.0 - preds)) @ X + jnp.eye(dim) / alpha
        return 0.5 * (metric + metric.T)

    def empirical_fisher_metric_fn(theta):
        metric = jnp.eye(dim) / alpha

        def func(index):
            c_grad = jax.grad(
                lambda c_theta: jsps.bernoulli.logpmf(
                    y[index], jax.nn.sigmoid(jnp.dot(X[index, :], c_theta))
                )
            )(theta)
            return jnp.outer(c_grad, c_grad)

        outers = jax.vmap(func)(jnp.arange(X.shape[0]))
        metric = metric + jnp.sum(outers, axis=0)
        return 0.5 * (metric + metric.T)

    def fisher_christoffel_fn(theta, v):
        # inspired by ChatGPT
        preds = jax.nn.sigmoid(jnp.dot(X, theta))

        func = (
            lambda i: (X.T * (preds * (1.0 - preds) * (1.0 - 2.0 * preds) * X[:, i]))
            @ X
            @ v
        )
        entity = jnp.dot(jax.vmap(func, out_axes=1)(jnp.arange(dim)), v)
        result = jnp.linalg.solve(fisher_metric_fn(theta), 0.5 * entity)

        return result

    def empirical_fisher_christoffel_fn(theta, v):
        return christoffel_fn(empirical_fisher_metric_fn, theta, v)

    def grad_fn(theta):
        preds = jax.nn.sigmoid(jnp.dot(X, theta))
        grad = jnp.sum(X * jnp.expand_dims(y - preds, 1), axis=0) - theta / alpha
        return grad

    def bergamin_christoffel_fn(theta, v):
        preds = jax.nn.sigmoid(jnp.dot(X, theta))
        grad = jnp.sum(X * jnp.expand_dims(y - preds, 1), axis=0) - theta / alpha
        hvp_v = -((X @ v).T * preds * (1.0 - preds)) @ X @ v - jnp.dot(v, v) / alpha
        norm_grad_2 = jnp.dot(grad, grad)

        W_2 = 1.0 + norm_grad_2
        mho = hvp_v / W_2
        return mho * grad

    return (
        dim,
        logp_fn,
        fisher_metric_fn,
        fisher_christoffel_fn,
        empirical_fisher_christoffel_fn,
        grad_fn,
        bergamin_christoffel_fn,
    )


# https://rdrr.io/cran/MASS/man/synth.tr.html
# GPL-2 | GPL-3
def lr_ripley(standardized=False):
    return get_logistic_fn("data/ripley.npy", standardized=standardized)


# https://rdrr.io/cran/MASS/man/Pima.tr.html
# GPL-2 | GPL-3
def lr_pima(standardized=False):
    return get_logistic_fn("data/pima.npy", standardized=standardized)


# https://archive.ics.uci.edu/dataset/98/statlog+project
# CC BY 4.0
def lr_heart(standardized=False):
    return get_logistic_fn("data/heart.npy", standardized=standardized)


# https://archive.ics.uci.edu/dataset/98/statlog+project
# CC BY 4.0
def lr_australian(standardized=False):
    return get_logistic_fn("data/australian.npy", standardized=standardized)


# https://archive.ics.uci.edu/dataset/98/statlog+project
# CC BY 4.0
def lr_german(standardized=False):
    return get_logistic_fn("data/german.npy", standardized=standardized)


def funnel():
    dim = 2

    def logp_fn(theta):
        return jsps.norm.logpdf(theta[1], loc=0.0, scale=3.0) + jsps.norm.logpdf(
            theta[0], loc=0.0, scale=jnp.exp(0.5 * theta[1])
        )

    def hausdorff_logp_fn(theta):
        return logp_fn(theta) - 0.5 * jnp.linalg.slogdet(fisher_metric_fn(theta))[1]

    def fisher_metric_fn(theta):
        jacobian = jnp.array(
            [
                [
                    jnp.exp(-0.5 * theta[1]),
                    -0.5 * theta[0] * jnp.exp(-0.5 * theta[1]),
                ],
                [0.0, 1.0 / 3.0],
            ]
        )
        metric = jacobian.T @ jacobian
        return 0.5 * (metric + metric.T)

    def fisher_christoffel_fn(theta, v):
        return christoffel_fn(fisher_metric_fn, theta, v)

    return (
        dim,
        logp_fn,
        hausdorff_logp_fn,
        fisher_metric_fn,
        fisher_christoffel_fn,
    )
