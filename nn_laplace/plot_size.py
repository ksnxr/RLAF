import matplotlib.pyplot as plt
import torch
import numpy as np
import sacred
from pathlib import Path
import os
from utils import sneaky_artifact, get_regression_data

from sacred import Experiment

ex = Experiment("main")


@ex.config
def my_config():
    folder = "nn_reg_la_size_logs"


@ex.automain
def my_main(folder, _run):
    plt.rcParams["font.size"] = 16
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    (
        X_train,
        y_train,
        train_loader,
        X_plot,
        X_test,
        y_test,
        X_mean,
        X_std,
        y_mean,
        y_std,
    ) = get_regression_data(num_train=500, num_test=100, standardized=True)

    ax1.scatter(
        X_train.squeeze().detach().cpu().numpy() * X_std + X_mean,
        y_train.squeeze().detach().cpu().numpy() * y_std + y_mean,
        alpha=0.3,
        color="tab:orange",
    )

    ax1.set_xlim([-5.0, 13.0])
    ax1.set_ylim([-4.0, 4.0])

    num_trains = [100, 200, 500, 1000, 2000, 5000]
    num_hiddens = [5, 10, 20, 35]
    repeats = 5

    method_names = ["Bergamin", "Fisher"]
    plot_names = {"Bergamin": "RLA-B", "Fisher": "RLA-F"}

    all_num_evals = dict()
    all_times = dict()
    for method_name in method_names:
        all_num_evals[method_name] = dict()
        all_times[method_name] = dict()

    all_num_params = []

    for num_train in num_trains:
        if num_train not in all_num_evals:
            for method_name in method_names:
                all_num_evals[method_name][num_train] = dict()
                all_times[method_name][num_train] = dict()

        for num_hidden in num_hiddens:
            num_params = num_hidden * num_hidden + 4 * num_hidden + 1
            if num_params not in all_num_params:
                all_num_params.append(num_params)
            for repeat in range(repeats):
                for method_name in method_names:
                    try:
                        with open(
                            f"{folder}/{num_train}_{num_hidden}/{repeat+1}/1/num_evals/False_{method_name.lower()}_num_evals.pt",
                            "rb",
                        ) as f:
                            num_evals = torch.load(f).numpy()
                        with open(
                            f"{folder}/{num_train}_{num_hidden}/{repeat+1}/1/times/False_{method_name.lower()}_times.pt",
                            "rb",
                        ) as f:
                            times = torch.load(f).numpy()

                        if num_params not in all_num_evals[method_name][num_train]:
                            all_num_evals[method_name][num_train][
                                num_params
                            ] = num_evals
                            all_times[method_name][num_train][num_params] = times
                        else:
                            all_num_evals[method_name][num_train][num_params] = (
                                np.concatenate(
                                    (
                                        all_num_evals[method_name][num_train][
                                            num_params
                                        ],
                                        num_evals,
                                    ),
                                    axis=0,
                                )
                            )
                            all_times[method_name][num_train][num_params] = (
                                np.concatenate(
                                    (
                                        all_times[method_name][num_train][num_params],
                                        times,
                                    ),
                                    axis=0,
                                )
                            )
                    except:
                        pass

    for method_name in method_names:
        for num_params in all_num_params:
            current_num_evals = []
            current_times = []
            for num_train in num_trains:
                current_num_evals.append(
                    np.mean(all_num_evals[method_name][num_train][num_params])
                )
                current_times.append(
                    np.mean(all_times[method_name][num_train][num_params])
                )

            # plt.plot(num_trains, current_num_evals, label=f"{num_params} params, {method_name}")
            ax2.plot(
                num_trains,
                current_times,
                label=f"{num_params} params, {plot_names[method_name]}",
            )
    # https://stackoverflow.com/a/39473158
    plt.legend()
    ax2.set_xlabel("Number of data points")
    ax2.set_ylabel("Number of seconds per sample")

    # https://matplotlib.org/stable/users/explain/axes/axes_scales.html
    ax2.set_xscale("log")

    plt.tight_layout()
    file_name = sneaky_artifact(_run, "figs", "scalability.png")
    plt.savefig(file_name, dpi=200, bbox_inches="tight")
