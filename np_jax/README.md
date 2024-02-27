# Environment

We use [Sacred](https://github.com/IDSIA/sacred) to record the experiments. The Python version is `3.11.5`. The main dependencies are:
* jax==0.4.7
* diffrax
* matplotlib
* scipy
* pandas
* pot
* tqdm
* cmdstan
* scikit-lern
* cmdstanpy
* sacred
* numpy==1.23.5
* sacred==0.8.4
* loky

# Usage

## Obtaining MAP estimate and ground truth samples

Run `get_quantitities.py`.

An example command is as follows
```
python get_quantities.py -F logs with model=banana save_figures=True
```
where `logs` is the directory to save logs and other generated quantities.

The other options can be found in function `def my_config()` in file `get_quantities.py`. We describe the available options as follows
1. `model`: which model to run,
2. `standardized`: only apply to logistic regression experiments, whether to apply standardization
4. `save_figures`: whether to save figures (figures are saved to `np_jax/figs` directory),
5. `show_progress`: whether to show Stan sampling progress, if using Stan to obtain the samples.

The results contain 3 values, 
1. the found MAP estimate `MAP`, 
2. Stan diagnose, if the ground truth samples are generated using Stan `diagnose`, 
3. in case of using Stan to generate the samples, whether "Processing complete, no problems detected." is found inside Stan diagnose `no_problems`, i.e. whether Stan diagnoses conclude that there are no problems when generating the samples; in case of analytical samples, it is always `True`.

## Obtaining LA samples

Run `samples.py`.

An example command is as follows
```
python samples.py -F logs with model=banana save_figures=True
```
where `logs` is the directory to save logs and other generated quantities.

The other options can be found in function `def my_config()` in file `samples.py`. We describe the available options as follows
1. `model`: which model to run,
2. `num_samples`: how many samples to obtain,
3. `euclidean`: whether to run ELA,
4. `bergamin`: whether to run RLA-B,
5. `monge`: whether to run RLA-BLog,
6. `fisher`: whether to run RLA-F,
7. `empirical_fisher`: whether to run RLA-F with Empirical Fisher in place of Fisher,
8. `save_figures`: whether to save figures,
9. `save_samples`: whether to save samples,
10. `use_diffrax`: if True, use Diffrax for integration; else use SciPy for integration,
11. `standardized`: only apply to logistic regression experiments, whether to apply standardization,
13. `run_hessian_precision`: whether to run Laplace variant with negative Hessian as precision,
14. `run_fisher_precision`: whether to run Laplace variant with Fisher as precision,
15. `calc_metric`: whether to calculate Wasserstein distance to the ground truth samples.

## Generating plots

### Figure 1

Run `get_geodesic_plots.py`.

> Note: requires at least having run `get_quantities.py` for model `banana_hausdorff`

An example command is as follows
```
python get_geodesic_plots.py -F logs
```
where `logs` is the directory to save logs and other generated quantities.

The other options can be found in function `def my_config()` in file `get_geodesic_plots.py`. We describe the available options as follows
1. `num_samples`: how many samples to obtain,
2. `calc_metric`: whether to calculate Wasserstein distance to the ground truth samples (require having run `get_quantities.py` for model `banana_hausdorff`),
3. `seed`: random seed for generating samples

### Figure 3

Run `bias_dim.py`.

An example command is as follows
```
python bias_dim.py -F logs
```
where `logs` is the directory to save logs and other generated quantities.

The other options can be found in function `def my_config()` in file `bias_dim.py`. We describe the available options as follows
1. `euclidean`: whether to run ELA,
2. `bergamin`: whether to run RLA-B,
3. `repeats`: number of independent runs to repeat

### First subplot of Figure 5

Run `get_banana_plots.py`.

> Note: requires at least having run `get_quantities.py` for model `banana_hausdorff`

An example command is as follows
```
python get_banana_plots.py -F logs
```
where `logs` is the directory to save logs and other generated quantities.

## Benchmark running time

Run `benchmark_time.py`.

> Note: requires having run `get_quantities.py` for the respective model

An example command is as follows
```
python benchmark_time.py -F logs with model=lr_ripley
```
where `logs` is the directory to save logs and other generated quantities.

The other options can be found in function `def my_config()` in file `benchmark_time.py`. We describe the available options as follows
1. `model`: which model to run,
2. `num_samples`: how many samples to obtain,
3. `bergamin`: whether to run RLA-B,
4. `monge`: whether to run RLA-BLog,
5. `fisher`: whether to run RLA-F,
6. `standardized`: only apply to logistic regression experiments, whether to apply standardization,
7. `run_hessian_precision`: whether to run LA variant with negative Hessian as precision,
8. `run_fisher_precision`: whether to run LA variant with Fisher as precision,
9. `calc_metric`: whether to calculate Wasserstein distance to the ground truth samples,
10. `save_samples`: whether to save the obtained samples,
11. `save_times`: whether to save the running times.

# Minor notes
1. We used `jax.numpy.linalg.solve` for calculating the inverse of the Fisher metric times a vector. However, using `jax.scipy.linalg.solve` can be even faster since we can benefit from the structure of the metric by setting `assume_a="pos"`.
