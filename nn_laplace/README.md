# Environment

We use [Sacred](https://github.com/IDSIA/sacred) to record the experiments. The Python version is `3.8.17`. The main dependencies are:
* matplotlib==3.7.3
* scipy==1.10.1
* cmdstanpy==1.1.0
* sacred==0.8.2
* torch==2.0.1
* numpy==1.23.5
* nngeometry==0.3
* scikit-learn==1.3.1

Additionally, the package [Laplace](https://github.com/AlexImmer/Laplace/, git commit id: f6af73668870834eee15893c27a250c71966495c) needs to be downloaded and installed.

# Usage

First, run `python get_snelson_data.py` inside directory `riemannian_laplace/nn_laplace/snelson_data`.

## Obtaining MAP estimate and LA samples

Run `regression.py`.

An example command is as follows
```
python regression.py -F logs with between=False
```
where `logs` is the directory to save logs and other generated quantities.

The other options can be found in function `def my_config()` in file `regression.py`. We describe the available options as follows
1. `laplaces`: whether to run different Laplace variants (if `True`, among others, `ELA` is run by default),
2. `bergamin`: whether to run `RLA-B`,
3. `monge`: whether to run `RLA-BLog`,
4. `fisher`: whether to run `RLA-F`,
5. `fisher_exp`: whether to run the `RLA-F` with inverse of the metric given by `nngeometry`,
6. `num_epochs`: number of epochs to train,
7. `num_samples`: how many samples to obtain,
8. `between`: whether to use data points between 1.5 and 3 as test set,
9. `deterministic_map`: whether to use deterministic MAP estimate (found by fixing seed),
10. `save_samples`: whether to save samples,
11. `lr`: learning rate of MAP training,
12. `weight_decay`: weight decay of MAP training,
13. `representation`: representation to be used for nngeometry (only applies to `fisher_exp`),
14. `standardized`: whether to use standardized data.

> `regression_size.py` is similar, with the main difference being using a data generation process and allowing to specify the number of hidden units using `num_hidden`.

The following are commands that could reproduce the results as reported in the paper (omitting setting logging directories etc.):

Snelson data, original data: (the other results for this can be reproduced by changing which algorithm to run and whether to set `between` to `True`, while keeping other settings the same)
```
python regression.py with bergamin=True monge=False fisher=True between=False deterministic_map=True save_samples=True num_samples=500
```

Snelson data, standardized data: (the other results for this can be reproduced by changing which algorithm to run and whether to set `between` to `True`, while keeping other settings the same)
```
python regression.py with bergamin=True monge=False fisher=True between=False deterministic_map=True num_samples=500 standardized=True weight_decay=1e-4
```

Scalability experiment: (the other results for this can be reproduced by changing which algorithm to run and `num_train`, `num_hidden`, while keeping other settings the same)
```
python regression_size.py with bergamin=True monge=False fisher=True deterministic_map=True num_samples=30 num_train=100 num_hidden=5 lr=1e-3 weight_decay=1e-4 num_epochs=50000 standardized=True
```

## Obtaining Stan NUTS samples

Run `get_stan_nuts_samples.py`.

An example command is as follows
```
python get_stan_nuts_samples.py -F logs with laplace_run_dir="logs/1"
```
where `logs` is the directory to save logs and other generated quantities.

The other options can be found in function `def my_config()` in file `get_stan_nuts.py`. We describe the available options as follows
1. `show_progress`: whether to show progress for Stan sampling,
2. `samples_path`: if given and is a file, calculate the evaluation metrics and generate the plots using the sub samples,
3. `max_treedepth`: specify maximum treedepth for Stan NUTS sampler,
4. `adapt_delta`: specify target acceptance probability for Stan NUTS sampler,
5. `classification`: whether the run is classification (raise `NotImplementedError` when set to `True`),
6. `repeats`: number of runs to repeat using the sub samples,
7. `laplace_run_dir`: run directory of the results of `regression.py`,
8. `sub_samples_size`: number of sub samples to use for calculating the evaluation metrics and generating the plots.

## Running times under varying scenarios

Run `plot_size.py`.
> Note: requires having obtained the logs under varying scenarios

An example command is as follows
```
python plot_size.py -F logs
```
where `logs` is the directory to save logs and other generated quantities.

The other options can be found in function `def my_config()` in file `plot_size.py`. We describe the available option as follows
1. `folder`: the path to the directory containing the logs.

# Minor notes
1. We directly used `torch.linalg.solve` for calculating the inverse of the Fisher metric times a vector. Since the metric is positive definite, using `torch.linalg.cholesky` and `torch.cholesky_solve` may be better.
