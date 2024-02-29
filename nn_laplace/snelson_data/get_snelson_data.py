import torch
import pickle
import numpy as np


# https://www.gatsby.ucl.ac.uk/~snelson/SPGP_dist.zip
# https://github.com/ratschlab/bnn_priors/blob/main/bnn_priors/data/snelson.npz
# MIT
def get_data(train_indexes, test_indexes, save_name, standardized):
    # https://stackoverflow.com/a/51465553
    data = np.load("snelson.npz")
    inputs = data["x_train"]
    outputs = data["y_train"]

    train_inputs = inputs[train_indexes]
    train_outputs = outputs[train_indexes]
    test_inputs = inputs[test_indexes]
    test_outputs = outputs[test_indexes]

    if not standardized:
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

    else:
        X_mean = np.mean(train_inputs)
        X_std = np.std(train_inputs)
        train_inputs = (train_inputs - X_mean) / X_std
        test_inputs = (test_inputs - X_mean) / X_std

        y_mean = np.mean(train_outputs)
        y_std = np.std(train_outputs)
        train_outputs = (train_outputs - y_mean) / y_std

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
        # https://stackoverflow.com/a/11389998
        snelson_data["X_mean"] = X_mean.item()
        snelson_data["X_std"] = X_std.item()
        snelson_data["y_mean"] = y_mean.item()
        snelson_data["y_std"] = y_std.item()

    with open(f"{save_name}.pkl", "wb") as f:
        pickle.dump(snelson_data, f)


np.random.seed(1)
test_indexes = np.random.choice(200, 50, replace=False)
train_indexes = np.asarray([i for i in range(200) if i not in test_indexes])

get_data(train_indexes, test_indexes, "snelson_data", standardized=False)
get_data(train_indexes, test_indexes, "snelson_data_standardized", standardized=True)

# https://stackoverflow.com/a/51465553
data = np.load("snelson.npz")
inputs = data["x_train"]

# with help from ChatGPT
test_indexes = np.asarray((1.5 <= inputs) & (inputs <= 3.0)).nonzero()[0]
train_indexes = np.asarray([i for i in range(200) if i not in test_indexes])

get_data(train_indexes, test_indexes, "snelson_data_between", standardized=False)
get_data(
    train_indexes, test_indexes, "snelson_data_between_standardized", standardized=True
)
