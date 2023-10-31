# https://github.com/aleximmer/Laplace/blob/main/examples/regression_example.md
import numpy as np
import torch
from sacred import Experiment

from utils import (
    get_snelson_data,
    get_snelson_data_test,
    plot_regression,
    sneaky_artifact,
    eval_regression,
    get_show_function,
)
from approximations.my_full import MyFullLaplace
from approximations.bergamin import BergaminLaplace
from approximations.monge import MongeLaplace
from approximations.fisher import FisherLaplace
from approximations.fisher_exp import FisherExpLaplace

ex = Experiment("main")


@ex.config
def my_config():
    laplaces = True
    bergamin = True
    monge = False
    fisher = True
    fisher_exp = False
    num_epochs = 20000
    num_samples = 100
    between = False
    deterministic_map = False
    save_samples = False
    lr = 1e-2
    weight_decay = 1e-5
    representation = "dense"


@ex.automain
def my_main(
    laplaces,
    bergamin,
    monge,
    fisher,
    fisher_exp,
    num_epochs,
    num_samples,
    between,
    deterministic_map,
    save_samples,
    lr,
    weight_decay,
    representation,
    _log,
    _run,
    _seed,
):
    show = get_show_function(_log)
    X_train, y_train, train_loader, X_plot = get_snelson_data(between=between)

    # deterministic MAP
    if deterministic_map:
        torch.manual_seed(1)
    else:
        torch.manual_seed(_seed)

    def get_model():
        return torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 1),
        )

    model = get_model()

    # train MAP
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(num_epochs):
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

    model.eval()
    results = dict()
    show("MAP")

    f_map = model(X_plot).squeeze().detach().cpu().numpy()
    plot_regression(
        X_train=X_train,
        y_train=y_train,
        X_plot=X_plot,
        model=model,
        between=between,
        samples=None,
        f_mu=f_map,
        pred_std=None,
        file_name=f"{between}_MAP",
        run=_run,
    )

    if save_samples:
        torch.save(
            model.state_dict(),
            sneaky_artifact(_run, "samples", f"{between}_map.pt"),
        )

    if laplaces:
        la = MyFullLaplace(model, "regression")
        la.fit(train_loader)
        log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(
            1, requires_grad=True
        )
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
        for _ in range(num_epochs):
            hyper_optimizer.zero_grad()
            neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
            neg_marglik.backward()
            hyper_optimizer.step()

        show(
            f"sigma={la.sigma_noise.item():.2f}, prior precision={la.prior_precision.item():.2f}",
        )
        results["sigma_noise"] = la.sigma_noise.item()
        results["prior_precision"] = la.prior_precision.item()

        mse, nll = eval_regression(
            get_snelson_data_test,
            model,
            torch.unsqueeze(la.mean, dim=0),
            la.sigma_noise.item(),
            between,
            _log,
        )
        results["MAP"] = {"MSE": mse, "NLL": nll}
        show("Laplace")

        sigma_noise = la.sigma_noise
        prior_precision = la.prior_precision

        # "stochastic" sampling
        torch.manual_seed(_seed)

        f_mu, f_var = la(X_plot, pred_type="glm")

        f_mu = f_mu.squeeze().detach().cpu().numpy()
        f_sigma = f_var.squeeze().detach().sqrt().cpu().numpy()
        pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)

        plot_regression(
            X_train=X_train,
            y_train=y_train,
            X_plot=X_plot,
            model=model,
            between=between,
            samples=None,
            f_mu=f_mu,
            pred_std=pred_std,
            file_name=f"{between}_linearized",
            run=_run,
        )

        samples, f_mu, f_var = la(
            X_plot, pred_type="nn", link_approx="mc", n_samples=num_samples
        )
        if save_samples:
            torch.save(
                samples, sneaky_artifact(_run, "samples", f"{between}_full_samples.pt")
            )

        mse, nll = eval_regression(
            get_snelson_data_test,
            model,
            samples,
            la.sigma_noise.item(),
            between,
            _log,
        )
        results["Full"] = {"MSE": mse, "NLL": nll}

        f_mu = f_mu.squeeze().detach().cpu().numpy()
        f_sigma = f_var.squeeze().detach().sqrt().cpu().numpy()
        pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)

        plot_regression(
            X_train=X_train,
            y_train=y_train,
            X_plot=X_plot,
            model=model,
            between=between,
            samples=samples,
            f_mu=f_mu,
            pred_std=pred_std,
            file_name=f"{between}_full_nn",
            run=_run,
        )

        def get_geometric_result(flag, name, la_class):
            if flag:
                show(name.capitalize())
                if name != "fisher_exp":
                    la = la_class(
                        model,
                        "regression",
                        sigma_noise=sigma_noise,
                        prior_precision=prior_precision,
                        X_train=X_train,
                        y_train=y_train,
                        logger=_log,
                    )
                else:
                    la = la_class(
                        model,
                        "regression",
                        sigma_noise=sigma_noise,
                        prior_precision=prior_precision,
                        X_train=X_train,
                        y_train=y_train,
                        logger=_log,
                        train_loader=train_loader,
                        representation=representation,
                    )
                la.fit(train_loader)

                show(
                    f"sigma={la.sigma_noise.item():.2f}, prior precision={la.prior_precision.item():.2f}",
                )

                samples, f_mu, f_var, num_evals, times = la(
                    X_plot, n_samples=num_samples
                )
                mean_num_evals = torch.mean(num_evals).item()
                std_num_evals = torch.std(num_evals).item()
                show(
                    f"number of evaluations: mean: {mean_num_evals}, std: {std_num_evals}"
                )
                mean_times = torch.mean(times).item()
                std_times = torch.std(times).item()
                show(f"running time: mean: {mean_times}, std: {std_times}")
                if save_samples:
                    torch.save(
                        samples,
                        sneaky_artifact(
                            _run, "samples", f"{between}_{name}_samples.pt"
                        ),
                    )
                    torch.save(
                        num_evals,
                        sneaky_artifact(
                            _run, "num_evals", f"{between}_{name}_num_evals.pt"
                        ),
                    )
                    torch.save(
                        times,
                        sneaky_artifact(_run, "times", f"{between}_{name}_times.pt"),
                    )

                mse, nll = eval_regression(
                    get_snelson_data_test,
                    model,
                    samples,
                    la.sigma_noise.item(),
                    between,
                    _log,
                )
                results[name.capitalize()] = {
                    "MSE": mse,
                    "NLL": nll,
                    "num_evals": [mean_num_evals, std_num_evals],
                    "times": [mean_times, std_times],
                }

                f_mu = f_mu.squeeze().detach().cpu().numpy()
                f_sigma = f_var.squeeze().detach().sqrt().cpu().numpy()
                pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item() ** 2)

                plot_regression(
                    X_train=X_train,
                    y_train=y_train,
                    X_plot=X_plot,
                    model=model,
                    between=between,
                    samples=samples,
                    f_mu=f_mu,
                    pred_std=pred_std,
                    file_name=f"{between}_{name}",
                    run=_run,
                )

        get_geometric_result(bergamin, "bergamin", BergaminLaplace)

        get_geometric_result(monge, "monge", MongeLaplace)

        get_geometric_result(fisher, "fisher", FisherLaplace)

        get_geometric_result(fisher_exp, "fisher_exp", FisherExpLaplace)

    return results
