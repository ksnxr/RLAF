from cmdstanpy import CmdStanModel
from utils import (
    get_snelson_data,
    get_snelson_data_test,
    sneaky_artifact,
    get_show_function,
    plot_regression,
    eval_regression,
)
import numpy as np
from sacred import Experiment
import os
import json
import math
import torch

ex = Experiment("main")

# ChatGPT
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


def get_regression_nuts_samples(
    subdir=None,
    show_progress=False,
    samples_path=None,
    max_treedepth=None,
    adapt_delta=None,
    repeats=None,
    sub_samples_size=100,
    run=None,
    laplace_run_dir=os.path.join(current_directory, "regression_logs/1"),
    logger=None,
    seed=None,
):
    show = get_show_function(logger)
    with open(os.path.join(laplace_run_dir, "config.json"), "rb") as f:
        between = json.load(f)["between"]
    with open(os.path.join(laplace_run_dir, "run.json"), "rb") as f:
        results = json.load(f)["result"]
    sigma_noise = results["sigma_noise"]
    prior_std = math.sqrt(1.0 / results["prior_precision"])
    show(f"sigma noise: {sigma_noise}, prior std: {prior_std}")

    show = get_show_function(logger)
    if samples_path is None:
        samples_path = f"{between}_stan_nuts_samples"
    if os.path.exists(samples_path):
        samples = np.load(samples_path)
    else:
        X, y, _, _ = get_snelson_data(between=between)
        N = X.shape[0]

        data = dict(
            N=N,
            x=X.numpy(),
            y=np.squeeze(y.numpy()),
            sigma_noise=sigma_noise,
            prior_std=prior_std,
        )

        # with help from ChatGPT
        model = CmdStanModel(
            stan_file=os.path.join(current_directory, "stan_models/regression.stan")
        )

        # default options from posteriordb
        if max_treedepth is None and adapt_delta is None:
            fit = model.sample(
                data=data,
                thin=10,
                chains=10,
                iter_warmup=10000,
                iter_sampling=20000,
                show_progress=show_progress,
                # fix seed for reproducibility
                seed=1,
            )
        elif max_treedepth is None and adapt_delta is not None:
            fit = model.sample(
                data=data,
                thin=10,
                chains=10,
                iter_warmup=10000,
                iter_sampling=20000,
                adapt_delta=adapt_delta,
                show_progress=show_progress,
                # fix seed for reproducibility
                seed=1,
            )
        elif max_treedepth is not None and adapt_delta is None:
            fit = model.sample(
                data=data,
                thin=10,
                chains=10,
                iter_warmup=10000,
                iter_sampling=20000,
                max_treedepth=max_treedepth,
                show_progress=show_progress,
                # fix seed for reproducibility
                seed=1,
            )
        else:
            fit = model.sample(
                data=data,
                thin=10,
                chains=10,
                iter_warmup=10000,
                iter_sampling=20000,
                max_treedepth=max_treedepth,
                adapt_delta=adapt_delta,
                show_progress=show_progress,
                # fix seed for reproducibility
                seed=1,
            )
        show(fit.summary())
        show(fit.diagnose())

        param_names = []
        for j in range(10):
            param_names.append(f"W1[1,{j+1}]")
        for j in range(10):
            param_names.append(f"b1[{j+1}]")
        for j in range(10):
            param_names.append(f"W2[{j+1},1]")
        param_names.append("b2")
        samples = fit.draws_pd()[param_names].to_numpy()
        np.save(sneaky_artifact(run, subdir, samples_path), samples)

    X_train, y_train, _, X_plot = get_snelson_data(between=between)

    def get_model():
        return torch.nn.Sequential(
            torch.nn.Linear(1, 10), torch.nn.Tanh(), torch.nn.Linear(10, 1)
        )

    model = get_model()
    model.eval()

    results = dict()

    np.random.seed(seed)
    for repeat in range(repeats):
        samples_idxes = np.random.choice(
            [i for i in range(samples.shape[0])], size=sub_samples_size, replace=False
        )
        sub_samples = torch.as_tensor(samples[samples_idxes, :], dtype=torch.float32)

        plot_regression(
            X_train=X_train,
            y_train=y_train,
            X_plot=X_plot,
            model=model,
            samples=sub_samples,
            f_mu=None,
            pred_std=None,
            file_name=f"{between}_stan_nuts_samples_{repeat}",
            calc_fs=True,
            sigma_noise=sigma_noise,
            run=run,
        )

        mse, nll = eval_regression(
            get_snelson_data_test,
            model,
            sub_samples,
            sigma_noise,
            between,
            logger,
        )
        results[str(repeat)] = {"MSE": mse, "NLL": nll}

    return results


@ex.config
def my_config():
    show_progress = False
    samples_path = None
    max_treedepth = None
    adapt_delta = None
    classification = False
    repeats = 5
    laplace_run_dir = "logs/nn_reg_lp_logs/True/1"
    sub_samples_size = 100


@ex.automain
def my_main(
    show_progress,
    samples_path,
    max_treedepth,
    adapt_delta,
    classification,
    repeats,
    laplace_run_dir,
    sub_samples_size,
    _run,
    _log,
    _seed,
):
    if classification:
        raise NotImplementedError
    else:
        return get_regression_nuts_samples(
            subdir="stan_nuts_samples",
            show_progress=show_progress,
            samples_path=samples_path,
            max_treedepth=max_treedepth,
            adapt_delta=adapt_delta,
            repeats=repeats,
            laplace_run_dir=laplace_run_dir,
            sub_samples_size=sub_samples_size,
            run=_run,
            logger=_log,
            seed=_seed,
        )
