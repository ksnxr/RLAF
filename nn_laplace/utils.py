import torch
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import pickle
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import sacred
from pathlib import Path
import os
from torch.distributions.normal import Normal
import math
import numpy as np
from scipy.integrate import solve_ivp

# ChatGPT
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


def get_christoffel_fun(dim, christoffel_fn):
    def func(t, y):
        theta = y[:dim]
        v = y[dim:]
        a = -christoffel_fn(theta, v)

        return np.concatenate([v, a])

    return func


def geodesic(dim, christoffel_fn, theta, v, t_span=(0.0, 1.0)):
    fun = get_christoffel_fun(dim=dim, christoffel_fn=christoffel_fn)
    return solve_ivp(
        fun=fun,
        t_span=t_span,
        y0=np.concatenate([theta, v]),
    )


def get_show_function(logger):
    if logger is not None:
        return logger.info
    else:
        return print


# see https://pytorch.org/docs/stable/_modules/torch/nn/utils/convert_parameters.html
def get_converter_functions(model):
    numels = dict()
    shapes = dict()
    for name, param in dict(model.named_parameters()).items():
        numels[name] = param.numel()
        shapes[name] = param.size()

    def param_shape_to_vector(params):
        vec = []
        for param in params.values():
            vec.append(param.view(-1))
        return torch.cat(vec)

    def vector_to_param_shape(vector):
        params = dict()
        count = 0
        for name in numels.keys():
            new_count = count + numels[name]
            params[name] = vector[count:new_count].view(shapes[name])
            count = new_count
        return params

    return param_shape_to_vector, vector_to_param_shape


# evaluation inspired by bnn_priors


# https://github.com/ratschlab/bnn_priors/blob/3597cf45a0c2496dd9e053090b3786f9fae573bb/bnn_priors/exp_utils.py#L300C5-L301C66
def _log_space_mean(tensor, dim):
    return tensor.logsumexp(dim) - math.log(tensor.size(dim))


@torch.no_grad()
def eval_regression(get_data_fn, model, samples, noise_std, between, standardized, logger=None):
    show = get_show_function(logger)
    if not standardized:
        X_test, y_test = get_data_fn(between, standardized=False)
    else:
        X_test, y_test, y_mean, y_std = get_data_fn(between, standardized=True)
        noise_std = noise_std * y_std
    final_preds = None
    num_samples = samples.shape[0]
    log_probs = torch.zeros((num_samples, y_test.shape[0]))
    mean = parameters_to_vector(model.parameters())
    for i in range(num_samples):
        sample = samples[i, :]
        vector_to_parameters(sample, model.parameters())
        if not standardized:
            model_pred = model(X_test)
        else:
            model_pred = model(X_test) * y_std + y_mean
        if final_preds is None:
            final_preds = model_pred
        else:
            final_preds += model_pred

        c_log_probs = (
            Normal(
                loc=model_pred,
                scale=torch.full_like(model_pred, noise_std),
            )
            .log_prob(y_test)
            .squeeze()
        )
        log_probs[i, :] = c_log_probs
    vector_to_parameters(mean, model.parameters())
    mse = torch.nn.MSELoss()(final_preds / float(num_samples), y_test).detach().item()
    nll = -_log_space_mean(log_probs, 0).mean().detach().item()
    show(f"MSE loss: {mse}, NLL: {nll}")
    return mse, nll


@torch.no_grad()
def eval_regression_data(X_test, y_test, model, samples, noise_std, standardized, y_mean, y_std, logger=None):
    show = get_show_function(logger)
    if standardized:
        noise_std = noise_std * y_std
    final_preds = None
    num_samples = samples.shape[0]
    log_probs = torch.zeros((num_samples, y_test.shape[0]))
    mean = parameters_to_vector(model.parameters())
    for i in range(num_samples):
        sample = samples[i, :]
        vector_to_parameters(sample, model.parameters())
        if not standardized:
            model_pred = model(X_test)
        else:
            model_pred = model(X_test) * y_std + y_mean
        if final_preds is None:
            final_preds = model_pred
        else:
            final_preds += model_pred

        c_log_probs = (
            Normal(
                loc=model_pred,
                scale=torch.full_like(model_pred, noise_std),
            )
            .log_prob(y_test)
            .squeeze()
        )
        log_probs[i, :] = c_log_probs
    vector_to_parameters(mean, model.parameters())
    mse = torch.nn.MSELoss()(final_preds / float(num_samples), y_test).detach().item()
    nll = -_log_space_mean(log_probs, 0).mean().detach().item()
    show(f"MSE loss: {mse}, NLL: {nll}")
    return mse, nll


@torch.no_grad()
def eval_regression_random(get_data_fn, model, samples, noise_std, logger=None):
    show = get_show_function(logger)
    X_test, y_test = get_data_fn()
    final_preds = None
    num_samples = samples.shape[0]
    log_probs = torch.zeros((num_samples, y_test.shape[0]))
    mean = parameters_to_vector(model.parameters())
    for i in range(num_samples):
        sample = samples[i, :]
        vector_to_parameters(sample, model.parameters())
        model_pred = model(X_test)
        if final_preds is None:
            final_preds = model_pred
        else:
            final_preds += model_pred

        c_log_probs = (
            Normal(loc=model_pred, scale=torch.full_like(model_pred, noise_std))
            .log_prob(y_test)
            .squeeze()
        )
        log_probs[i, :] = c_log_probs
    vector_to_parameters(mean, model.parameters())
    mse = torch.nn.MSELoss()(final_preds / float(num_samples), y_test).detach().item()
    nll = -_log_space_mean(log_probs, 0).mean().detach().item()
    show(f"MSE loss: {mse}, NLL: {nll}")
    return mse, nll


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
        os.mkdir(subdir_path)

    return os.path.join(subdir_path, name)


def get_snelson_data_random():
    test_indexes = np.random.choice(200, 50, replace=False)
    train_indexes = np.asarray([i for i in range(200) if i not in test_indexes])

    inputs = []
    with open(os.path.join(current_directory, "snelson_data/train_inputs")) as f:
        for line in f:
            inputs.append(float(line))
    inputs = np.asarray(inputs)

    outputs = []
    with open(os.path.join(current_directory, "snelson_data/train_outputs")) as f:
        for line in f:
            outputs.append(float(line))
    outputs = np.asarray(outputs)

    train_inputs = inputs[train_indexes]
    train_outputs = outputs[train_indexes]
    test_inputs = inputs[test_indexes]
    test_outputs = outputs[test_indexes]

    snelson_data = dict()
    snelson_data["train_inputs"] = torch.tensor(
        train_inputs, dtype=torch.float32
    ).unsqueeze(-1)
    snelson_data["train_outputs"] = torch.tensor(
        train_outputs, dtype=torch.float32
    ).unsqueeze(-1)
    snelson_data["test_inputs"] = torch.tensor(
        test_inputs, dtype=torch.float32
    ).unsqueeze(-1)
    snelson_data["test_outputs"] = torch.tensor(
        test_outputs, dtype=torch.float32
    ).unsqueeze(-1)

    def get_snelson_data(batch_size=150):
        X_train = snelson_data["train_inputs"]
        y_train = snelson_data["train_outputs"]
        train_loader = data_utils.DataLoader(
            data_utils.TensorDataset(X_train, y_train), batch_size=batch_size
        )
        X_plot = torch.linspace(-0.5, 6.5, 500).unsqueeze(-1)
        return X_train, y_train, train_loader, X_plot

    def get_snelson_data_test():
        X_test = snelson_data["test_inputs"]
        y_test = snelson_data["test_outputs"]
        return X_test, y_test

    return get_snelson_data, get_snelson_data_test


def get_snelson_data(between=False, standardized=False):
    if not standardized:
        if between:
            file_name = "snelson_data/snelson_data_between.pkl"
        else:
            file_name = "snelson_data/snelson_data.pkl"

        with open(os.path.join(current_directory, file_name), "rb") as f:
            snelson_data = pickle.load(f)
        X_train = snelson_data["train_inputs"]
        y_train = snelson_data["train_outputs"]
        batch_size = X_train.shape[0]
        train_loader = data_utils.DataLoader(
            data_utils.TensorDataset(X_train, y_train), batch_size=batch_size
        )
        X_plot = torch.linspace(-0.5, 6.5, 500).unsqueeze(-1)
        return X_train, y_train, train_loader, X_plot
    else:
        if between:
            file_name = "snelson_data/snelson_data_between_standardized.pkl"
        else:
            file_name = "snelson_data/snelson_data_standardized.pkl"

        with open(os.path.join(current_directory, file_name), "rb") as f:
            snelson_data = pickle.load(f)
        X_train = snelson_data["train_inputs"]
        y_train = snelson_data["train_outputs"]
        X_mean = snelson_data["X_mean"]
        X_std = snelson_data["X_std"]
        y_mean = snelson_data["y_mean"]
        y_std = snelson_data["y_std"]
        batch_size = X_train.shape[0]
        train_loader = data_utils.DataLoader(
            data_utils.TensorDataset(X_train, y_train), batch_size=batch_size
        )
        X_plot = (torch.linspace(-0.5, 6.5, 500).unsqueeze(-1) - X_mean) / X_std
        return X_train, y_train, train_loader, X_plot, X_mean, X_std, y_mean, y_std


def get_snelson_data_test(between=False, standardized=False):
    if not standardized:
        if between:
            file_name = "snelson_data/snelson_data_between.pkl"
        else:
            file_name = "snelson_data/snelson_data.pkl"

        with open(os.path.join(current_directory, file_name), "rb") as f:
            snelson_data = pickle.load(f)
        X_test = snelson_data["test_inputs"]
        y_test = snelson_data["test_outputs"]
        return X_test, y_test
    
    else:
        if between:
            file_name = "snelson_data/snelson_data_between_standardized.pkl"
        else:
            file_name = "snelson_data/snelson_data_standardized.pkl"

        with open(os.path.join(current_directory, file_name), "rb") as f:
            snelson_data = pickle.load(f)
        X_test = snelson_data["test_inputs"]
        y_test = snelson_data["test_outputs"]
        y_mean = snelson_data["y_mean"]
        y_std = snelson_data["y_std"]
        return X_test, y_test, y_mean, y_std


# adapted from https://github.com/aleximmer/Laplace/blob/f6af73668870834eee15893c27a250c71966495c/examples/helper/dataloaders.py#L36
def get_regression_data(num_train=150, num_test=50, batch_size=128, sigma_noise=0.3, standardized=False):
    X = (torch.rand(num_train+num_test) * 8).unsqueeze(-1)
    y = torch.sin(X) + torch.randn_like(X) * sigma_noise
    X_train, y_train = X[:num_train, :], y[:num_train, :]
    X_test, y_test = X[num_train:, :], y[num_train:, :]
    
    if not standardized:
        train_loader = data_utils.DataLoader(
            data_utils.TensorDataset(X_train, y_train), 
            batch_size=batch_size
        )
        X_plot = torch.linspace(-5.0, 13.0, 500).unsqueeze(-1)
        return X_train, y_train, train_loader, X_plot, X_test, y_test
    
    else:

        # Normalize both X and y, with suggestion from ChatGPT

        X_mean = torch.mean(X_train)
        X_std = torch.std(X_train)
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

        y_mean = torch.mean(y_train)
        y_std = torch.std(y_train)
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        train_loader = data_utils.DataLoader(
            data_utils.TensorDataset(X_train, y_train), 
            batch_size=batch_size
        )
        X_plot = (torch.linspace(-5.0, 13.0, 500).unsqueeze(-1) - X_mean) / X_std
        return X_train, y_train, train_loader, X_plot, X_test, y_test, X_mean.item(), X_std.item(), y_mean.item(), y_std.item()


@torch.no_grad()
def plot_regression(
    X_train,
    y_train,
    X_plot,
    model,
    between=False,
    samples=None,
    subsampled=False,
    f_mu=None,
    pred_std=None,
    file_name="regression_example",
    calc_fs=False,
    sigma_noise=None,
    standardized=False,
    X_mean=None,
    X_std=None,
    y_mean=None,
    y_std=None,
    run=None,
):
    ylim = [-4.0, 6.0]
    if standardized:
        if f_mu is not None:
            f_mu = f_mu * y_std + y_mean
        if sigma_noise is not None:
            sigma_noise = sigma_noise * y_std

    plt.figure(figsize=(7, 5))
    plt.rcParams["font.size"] = 30
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"
    # plt.scatter(
    #     X_train.squeeze().detach().cpu().numpy(),
    #     y_train.squeeze().detach().cpu().numpy(),
    #     alpha=0.3,
    #     color="tab:orange",
    # )
    xs = X_train.squeeze().detach().cpu().numpy()
    ys = np.linspace(ylim[0], ylim[1], 10)
    if not standardized:
        x0 = np.min(xs)
        x1 = 1.5
        x2 = 3.0
        x3 = np.max(xs)
    else:
        x0 = np.min(xs) * X_std + X_mean
        x1 = 1.5
        x2 = 3.0
        x3 = np.max(xs) * X_std + X_mean

    if between:
        plt.fill_betweenx(ys, x0, x1, alpha=0.2, color="gray")
        plt.fill_betweenx(ys, x2, x3, alpha=0.2, color="gray")
    else:
        plt.fill_betweenx(ys, x0, x3, alpha=0.2, color="gray")

    if samples is not None:
        num_samples = samples.shape[0]
        if num_samples > 20 and subsampled:
            np.random.seed(1)
            plot_idxes = np.random.choice(
                np.asarray([i for i in range(num_samples)]), 50, replace=False
            )
        else:
            plot_idxes = np.asarray([i for i in range(num_samples)])
        fs = list()
        map_estimate = parameters_to_vector(model.parameters())
        for i in range(samples.shape[0]):
            if i in plot_idxes:
                sample = samples[i, :]
                vector_to_parameters(sample, model.parameters())
                if not standardized:
                    f = model(X_plot)
                else:
                    f = model(X_plot) * y_std + y_mean
                fs.append(f.detach())
                if not standardized:
                    plt.plot(
                        X_plot.squeeze().detach().cpu().numpy(),
                        f.squeeze().detach().cpu().numpy(),
                        linewidth=1.0,
                        alpha=0.3,
                        color="dodgerblue",
                    )
                else:
                    plt.plot(
                        X_plot.squeeze().detach().cpu().numpy() * X_std + X_mean,
                        f.squeeze().detach().cpu().numpy(),
                        linewidth=1.0,
                        alpha=0.3,
                        color="dodgerblue",
                    )
        fs = torch.stack(fs)
        vector_to_parameters(map_estimate, model.parameters())

    if calc_fs:
        assert samples is not None
        assert sigma_noise is not None
        if len(samples) == 1:
            raise Exception
        f_mu = fs.mean(dim=0).squeeze().detach().cpu().numpy()
        # f_sigma = fs.var(dim=0).squeeze().detach().sqrt().cpu().numpy()
        # sigma_noise = np.asarray(sigma_noise)
        # pred_std = np.sqrt(f_sigma**2 + sigma_noise**2)

    X_plot = X_plot.squeeze().detach().cpu().numpy()
    if standardized:
        X_plot = X_plot * X_std + X_mean
    plt.plot(
        X_plot,
        f_mu,
        label="$\mathbb{E}[f]$",
        linewidth=2.5,
        color="navy",
    )
    # if pred_std is not None:
    #     plt.fill_between(
    #         X_plot,
    #         f_mu - pred_std * 2,
    #         f_mu + pred_std * 2,
    #         alpha=0.3,
    #         color="tab:blue",
    #     )

    plt.ylim(ylim)
    plt.xlim([np.min(X_plot), np.max(X_plot)])

    # ChatGPT
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    if run is None:
        plt.savefig(f"figs/{file_name}.png", dpi=200, bbox_inches="tight")
    else:
        plt.savefig(
            sneaky_artifact(run, "figs", f"{file_name}.png"),
            dpi=200,
            bbox_inches="tight",
        )

@torch.no_grad()
def plot_regression_data(
    X_train,
    y_train,
    plot_data=False,
    X_plot=None,
    model=None,
    between=False,
    samples=None,
    subsampled=False,
    f_mu=None,
    pred_std=None,
    file_name="regression_example",
    calc_fs=False,
    sigma_noise=None,
    standardized=False,
    X_mean=None,
    X_std=None,
    y_mean=None,
    y_std=None,
    run=None,
):
    ylim = [-4.0, 4.0]
    if standardized:
        if f_mu is not None:
            f_mu = f_mu * y_std + y_mean
        if sigma_noise is not None:
            sigma_noise = sigma_noise * y_std

    plt.figure(figsize=(7, 5))
    plt.rcParams["font.size"] = 30
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"
    if plot_data:
        if not standardized:
            plt.scatter(
                X_train.squeeze().detach().cpu().numpy(),
                y_train.squeeze().detach().cpu().numpy(),
                alpha=0.3,
                color="tab:orange",
            )
        else:
            plt.scatter(
                X_train.squeeze().detach().cpu().numpy() * X_std + X_mean,
                y_train.squeeze().detach().cpu().numpy() * y_std + y_mean,
                alpha=0.3,
                color="tab:orange",
            )
        plt.ylim(ylim)
        plt.xlim([-5.0, 13.0])
    else:
        xs = X_train.squeeze().detach().cpu().numpy()
        ys = np.linspace(ylim[0], ylim[1], 10)
        if not standardized:
            plt.fill_betweenx(ys, np.min(xs), np.max(xs), alpha=0.2, color="gray")
        else:
            plt.fill_betweenx(ys, np.min(xs) * X_std + X_mean, np.max(xs) * X_std + X_mean, alpha=0.2, color="gray")

        if samples is not None:
            num_samples = samples.shape[0]
            if num_samples > 20 and subsampled:
                np.random.seed(1)
                plot_idxes = np.random.choice(
                    np.asarray([i for i in range(num_samples)]), 50, replace=False
                )
            else:
                plot_idxes = np.asarray([i for i in range(num_samples)])
            fs = list()
            map_estimate = parameters_to_vector(model.parameters())
            for i in range(samples.shape[0]):
                if i in plot_idxes:
                    sample = samples[i, :]
                    vector_to_parameters(sample, model.parameters())
                    if not standardized:
                        f = model(X_plot)
                    else:
                        f = model(X_plot) * y_std + y_mean
                    fs.append(f.detach())
                    if not standardized:
                        plt.plot(
                            X_plot.squeeze().detach().cpu().numpy(),
                            f.squeeze().detach().cpu().numpy(),
                            linewidth=1.0,
                            alpha=0.3,
                            color="dodgerblue",
                        )
                    else:
                        plt.plot(
                            X_plot.squeeze().detach().cpu().numpy() * X_std + X_mean,
                            f.squeeze().detach().cpu().numpy(),
                            linewidth=1.0,
                            alpha=0.3,
                            color="dodgerblue",
                        )
            fs = torch.stack(fs)
            vector_to_parameters(map_estimate, model.parameters())

        if calc_fs:
            assert samples is not None
            assert sigma_noise is not None
            if len(samples) == 1:
                raise Exception
            f_mu = fs.mean(dim=0).squeeze().detach().cpu().numpy()
            # f_sigma = fs.var(dim=0).squeeze().detach().sqrt().cpu().numpy()
            # sigma_noise = np.asarray(sigma_noise)
            # pred_std = np.sqrt(f_sigma**2 + sigma_noise**2)

        X_plot = X_plot.squeeze().detach().cpu().numpy()
        if standardized:
            X_plot = X_plot * X_std + X_mean
        plt.plot(
            X_plot,
            f_mu,
            label="$\mathbb{E}[f]$",
            linewidth=2.5,
            color="navy",
        )
        # if pred_std is not None:
        #     plt.fill_between(
        #         X_plot,
        #         f_mu - pred_std * 2,
        #         f_mu + pred_std * 2,
        #         alpha=0.3,
        #         color="tab:blue",
        #     )

        plt.ylim(ylim)
        plt.xlim([np.min(X_plot), np.max(X_plot)])

    # ChatGPT
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    if run is None:
        plt.savefig(f"figs/{file_name}.png", dpi=200, bbox_inches="tight")
    else:
        plt.savefig(
            sneaky_artifact(run, "figs", f"{file_name}.png"),
            dpi=200,
            bbox_inches="tight",
        )