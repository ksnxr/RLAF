from laplace.baselaplace import FullLaplace
import torch
import numpy as np
from scipy.integrate import solve_bvp
from torch.distributions import MultivariateNormal
from torch.nn.utils import vector_to_parameters
from utils import get_converter_functions, geodesic
import time


def get_monge_gaussian_fun(dim, precision):
    def func(t, y):
        theta = y[:dim, :]
        v = y[dim:, :]
        theta_grad = precision @ theta
        norm_theta_grad_2 = np.sum(np.square(theta_grad), axis=0)

        W_2 = 1.0 + norm_theta_grad_2
        mho = np.sum(v * (precision @ v), axis=0) / W_2

        return np.concatenate([v, -mho * theta_grad])

    return func


def get_velocity(dim, precision, v, t_steps=500):
    fun = get_monge_gaussian_fun(dim, precision)

    def bc(ya, yb):
        return np.concatenate([ya[:dim], yb[:dim] - v])

    t_steps = 500
    x = np.linspace(0.0, 1.0, t_steps)

    # ChatGPT
    tile_v = np.tile(v[:, np.newaxis], (1, t_steps))
    y = np.concatenate([tile_v * x.reshape((1, t_steps)), tile_v], axis=0)

    return solve_bvp(
        fun=fun,
        bc=bc,
        x=x,
        y=y,
    )


class MongeLaplace(FullLaplace):
    def __init__(
        self,
        model,
        likelihood,
        sigma_noise=1.0,
        prior_precision=1.0,
        prior_mean=0.0,
        temperature=1.0,
        enable_backprop=False,
        backend=None,
        backend_kwargs=None,
        X_train=None,
        y_train=None,
        logger=None,
    ):
        super().__init__(
            model,
            likelihood,
            sigma_noise,
            prior_precision,
            prior_mean,
            temperature,
            enable_backprop,
            backend,
            backend_kwargs,
        )
        self.X_train = X_train
        self.y_train = y_train
        param_shape_to_vector, vector_to_param_shape = get_converter_functions(
            self.model
        )
        self.param_shape_to_vector = param_shape_to_vector
        self.vector_to_param_shape = vector_to_param_shape
        self.dim = sum(p.numel() for p in self.model.parameters())
        self.logger = logger

    def regression_loss(self, vec_params):
        params = self.vector_to_param_shape(vec_params)
        prediction = torch.func.functional_call(self.model, params, (self.X_train,))
        loss = torch.sum(torch.square(self.y_train - prediction)) / torch.square(
            self.sigma_noise
        )
        loss += torch.sum(torch.square(vec_params)) * torch.squeeze(
            self.prior_precision
        )
        return 0.5 * loss

    def classification_loss(self, vec_params):
        params = self.vector_to_param_shape(vec_params)
        prediction = torch.func.functional_call(self.model, params, (self.X_train,))
        log_probs = torch.nn.functional.log_softmax(prediction, dim=1)
        loss = -torch.sum(log_probs[torch.arange(len(self.y_train)), self.y_train])
        loss += (
            0.5
            * torch.sum(torch.square(vec_params))
            * torch.squeeze(self.prior_precision)
        )
        return loss

    # https://pytorch.org/tutorials/intermediate/jacobians_hessians.html?utm_source=whats_new_tutorials&utm_medium=jacobians_hessians
    def regression_christoffel_fn(self, vec_theta, vec_v):
        vec_theta = torch.as_tensor(vec_theta, dtype=torch.float32, device=self._device)
        vec_v = torch.as_tensor(vec_v, dtype=torch.float32, device=self._device)
        vec_grad, vec_hvp = torch.func.jvp(
            torch.func.grad(self.regression_loss), (vec_theta,), (vec_v,)
        )

        W_2 = 1.0 + torch.dot(vec_grad, vec_grad)
        result = torch.dot(vec_v, vec_hvp) / W_2 * vec_grad
        return result.detach().cpu().numpy()

    # https://pytorch.org/tutorials/intermediate/jacobians_hessians.html?utm_source=whats_new_tutorials&utm_medium=jacobians_hessians
    def classification_christoffel_fn(self, vec_theta, vec_v):
        vec_theta = torch.as_tensor(vec_theta, dtype=torch.float32, device=self._device)
        vec_v = torch.as_tensor(vec_v, dtype=torch.float32, device=self._device)
        vec_grad, vec_hvp = torch.func.jvp(
            torch.func.grad(self.classification_loss), (vec_theta,), (vec_v,)
        )

        W_2 = 1.0 + torch.dot(vec_grad, vec_grad)
        result = torch.dot(vec_v, vec_hvp) / W_2 * vec_grad
        return result.detach().cpu().numpy()

    def _nn_predictive_samples(self, X, n_samples=100):
        fs = list()
        param_samples, num_evals, times = self.sample(n_samples)
        for sample in param_samples:
            vector_to_parameters(sample, self.model.parameters())
            f = self.model(X.to(self._device))
            fs.append(f.detach() if not self.enable_backprop else f)
        vector_to_parameters(self.mean, self.model.parameters())
        fs = torch.stack(fs)
        if self.likelihood == "classification":
            fs = torch.softmax(fs, dim=-1)
        return param_samples, fs, num_evals, times

    def __call__(
        self,
        x,
        n_samples=100,
    ):
        param_samples, samples, num_evals, times = self._nn_predictive_samples(
            x, n_samples
        )
        if self.likelihood == "regression":
            return (
                param_samples,
                samples.mean(dim=0),
                samples.var(dim=0),
                num_evals,
                times,
            )
        return param_samples, samples.mean(dim=0), num_evals, times

    def sample(self, n_samples=100):
        dist = MultivariateNormal(
            loc=torch.zeros_like(self.mean), scale_tril=self.posterior_scale
        )
        base_samples = dist.sample((n_samples,))
        final_samples = torch.zeros_like(base_samples)
        num_evals = torch.zeros(n_samples)
        times = torch.zeros(n_samples)

        if self.likelihood == "regression":
            for i in range(n_samples):
                # if i % 10 == 0:
                #     self.logger.info(i)
                base_sample = base_samples[i, :]

                t1 = time.time()
                velocity = get_velocity(
                    dim=self.dim,
                    precision=self.posterior_precision.detach().cpu().numpy(),
                    v=base_sample.squeeze().detach().cpu().numpy(),
                )["y"][self.dim :, 0]

                final_sample = geodesic(
                    dim=self.dim,
                    christoffel_fn=self.regression_christoffel_fn,
                    theta=self.mean.squeeze().detach().cpu().numpy(),
                    v=velocity.squeeze(),
                )
                times[i] = time.time() - t1

                final_samples[i, :] = torch.as_tensor(
                    final_sample["y"][: self.dim, -1], dtype=torch.float32
                )
                num_evals[i] = final_sample["nfev"]

        else:
            for i in range(n_samples):
                # if i % 10 == 0:
                #     self.logger.info(i)
                base_sample = base_samples[i, :]

                t1 = time.time()
                velocity = get_velocity(
                    dim=self.dim,
                    precision=self.posterior_precision.detach().cpu().numpy(),
                    v=base_sample.squeeze().detach().cpu().numpy(),
                )["y"][self.dim :, 0]

                final_sample = geodesic(
                    dim=self.dim,
                    christoffel_fn=self.classification_christoffel_fn,
                    theta=self.mean.squeeze().detach().cpu().numpy(),
                    v=velocity.squeeze(),
                )
                times[i] = time.time() - t1

                final_samples[i, :] = torch.as_tensor(
                    final_sample["y"][: self.dim, -1], dtype=torch.float32
                )
                num_evals[i] = final_sample["nfev"]

        return final_samples, num_evals, times
