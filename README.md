# Code for the paper Riemannian Laplace Approximation with the Fisher Metric

# Structure

* **np_jax** contains code for experiments other than neural network ones,
* **nn_laplace** contains code for neural network experiments.

For further details concerning each refer to the README.md inside each directory.

## Minor notes
ChatGPT was used as an assistance to write codes. Adopted suggestions, among others, e.g. for writing helper scripts, include
1. Use `metric = 0.5 * (metric + metric.T)` partly with suggestion from ChatGPT,
2. Use `torch.square` with suggestion from ChatGPT.
