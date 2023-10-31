from cmdstanpy import CmdStanModel
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from utils import get_show_function
import scipy.stats as sps

# ChatGPT
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


def get_banana_nuts_samples(
    samples_file=None,
    show_progress=False,
    logger=None,
):
    show = get_show_function(logger)
    ys = np.load(os.path.join(current_directory, "data/banana_ys.npy"))
    data = dict(y=ys)
    model = CmdStanModel(
        stan_file=os.path.join(current_directory, "stan_models/banana.stan")
    )
    fit = model.sample(
        data=data,
        thin=10,
        chains=10,
        iter_warmup=10000,
        iter_sampling=20000,
        adapt_delta=0.95,
        # fix seed for reproducibility
        seed=1,
        show_progress=show_progress,
    )
    show(fit.summary())
    show(fit.diagnose())

    if samples_file is not None:
        samples = fit.draws_pd()[["theta[1]", "theta[2]"]].to_numpy()
        np.save(samples_file, samples)

    return fit.diagnose()


def get_lr_nuts_samples(
    file_name,
    samples_file=None,
    standardized=False,
    show_progress=False,
    logger=None,
):
    show = get_show_function(logger)
    data = np.load(file_name)

    X = data[:, :-1]

    N = X.shape[0]

    if standardized:
        ss = StandardScaler()
        X = ss.fit_transform(X)

    new_x = np.ones((X.shape[0], 1))
    X = np.concatenate([X, new_x], axis=1)
    dim = X.shape[1]

    y = data[:, -1]
    # ChatGPT
    assert np.all((y == 0) | (y == 1))

    data = dict(N=N, D=dim, x=X, y=y.astype(int))

    model = CmdStanModel(
        stan_file=os.path.join(current_directory, "stan_models/lr.stan")
    )

    # default options from posteriordb
    fit = model.sample(
        data=data,
        thin=10,
        chains=10,
        iter_warmup=10000,
        iter_sampling=20000,
        # fix seed for reproducibility
        seed=1,
        show_progress=show_progress,
    )
    show(fit.summary())
    show(fit.diagnose())

    if samples_file is not None:
        param_names = [f"beta[{n + 1}]" for n in range(dim)]
        samples = fit.draws_pd()[param_names].to_numpy()
        np.save(samples_file, samples)

    return fit.diagnose()


def get_gaussian_samples(samples_file=None):
    np.random.seed(1)
    samples = sps.multivariate_normal.rvs(np.zeros(2), np.eye(2), 20000)
    if samples_file is not None:
        np.save(samples_file, samples)
    return ""


def get_squiggle_samples(samples_file=None, a=1.5, Sigma=np.eye(2)):
    np.random.seed(1)
    Sigma = np.asarray(Sigma)
    samples = sps.multivariate_normal.rvs(np.zeros(2), Sigma, 20000)
    samples[:, 1] = samples[:, 1] - np.sin(a * samples[:, 0])
    if samples_file is not None:
        np.save(samples_file, samples)
    return ""


def get_funnel_samples(samples_file=None):
    np.random.seed(1)
    x_raw = sps.norm.rvs(loc=0.0, scale=1.0, size=20000)
    y_raw = sps.norm.rvs(loc=0.0, scale=1.0, size=20000)
    y = 3.0 * y_raw
    x = np.exp(y / 2) * x_raw
    samples = np.stack([x, y], axis=1)
    if samples_file is not None:
        np.save(samples_file, samples)
    return ""


if __name__ == "__main__":
    pass
