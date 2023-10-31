# Modified based on https://github.com/aleximmer/Laplace
from laplace.baselaplace import FullLaplace
import torch
import numpy as np
from torch.nn.utils import vector_to_parameters


class MyFullLaplace(FullLaplace):
    def _nn_predictive_samples(self, X, n_samples=100):
        fs = list()
        param_samples = self.sample(n_samples)
        for sample in param_samples:
            vector_to_parameters(sample, self.model.parameters())
            f = self.model(X.to(self._device))
            fs.append(f.detach() if not self.enable_backprop else f)
        vector_to_parameters(self.mean, self.model.parameters())
        fs = torch.stack(fs)
        if self.likelihood == "classification":
            fs = torch.softmax(fs, dim=-1)
        return param_samples, fs

    def __call__(
        self,
        x,
        pred_type="glm",
        joint=False,
        link_approx="probit",
        n_samples=100,
        diagonal_output=False,
        generator=None,
    ):
        """Compute the posterior predictive on input data `x`.

        Parameters
        ----------
        x : torch.Tensor
            `(batch_size, input_shape)`

        pred_type : {'glm', 'nn'}, default='glm'
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictive. The GLM predictive is consistent with
            the curvature approximations used here.

        link_approx : {'mc', 'probit', 'bridge', 'bridge_norm'}
            how to approximate the classification link function for the `'glm'`.
            For `pred_type='nn'`, only 'mc' is possible.

        joint : bool
            Whether to output a joint predictive distribution in regression with
            `pred_type='glm'`. If set to `True`, the predictive distribution
            has the same form as GP posterior, i.e. N([f(x1), ...,f(xm)], Cov[f(x1), ..., f(xm)]).
            If `False`, then only outputs the marginal predictive distribution.
            Only available for regression and GLM predictive.

        n_samples : int
            number of samples for `link_approx='mc'`.

        diagonal_output : bool
            whether to use a diagonalized posterior predictive on the outputs.
            Only works for `pred_type='glm'` and `link_approx='mc'`.

        generator : torch.Generator, optional
            random number generator to control the samples (if sampling used).

        Returns
        -------
        predictive: torch.Tensor or Tuple[torch.Tensor]
            For `likelihood='classification'`, a torch.Tensor is returned with
            a distribution over classes (similar to a Softmax).
            For `likelihood='regression'`, a tuple of torch.Tensor is returned
            with the mean and the predictive variance.
            For `likelihood='regression'` and `joint=True`, a tuple of torch.Tensor
            is returned with the mean and the predictive covariance.
        """
        if pred_type not in ["glm", "nn"]:
            raise ValueError("Only glm and nn supported as prediction types.")

        if link_approx not in ["mc", "probit", "bridge", "bridge_norm"]:
            raise ValueError(f"Unsupported link approximation {link_approx}.")

        if pred_type == "nn" and link_approx != "mc":
            raise ValueError(
                "Only mc link approximation is supported for nn prediction type."
            )

        if generator is not None:
            if (
                not isinstance(generator, torch.Generator)
                or generator.device != x.device
            ):
                raise ValueError("Invalid random generator (check type and device).")

        if pred_type == "glm":
            f_mu, f_var = self._glm_predictive_distribution(
                x, joint=joint and self.likelihood == "regression"
            )
            # regression
            if self.likelihood == "regression":
                return f_mu, f_var
            # classification
            if link_approx == "mc":
                return self.predictive_samples(
                    x,
                    pred_type="glm",
                    n_samples=n_samples,
                    diagonal_output=diagonal_output,
                ).mean(dim=0)
            elif link_approx == "probit":
                kappa = 1 / torch.sqrt(1.0 + np.pi / 8 * f_var.diagonal(dim1=1, dim2=2))
                return torch.softmax(kappa * f_mu, dim=-1)
            elif "bridge" in link_approx:
                # zero mean correction
                f_mu -= (
                    f_var.sum(-1)
                    * f_mu.sum(-1).reshape(-1, 1)
                    / f_var.sum(dim=(1, 2)).reshape(-1, 1)
                )
                f_var -= torch.einsum(
                    "bi,bj->bij", f_var.sum(-1), f_var.sum(-2)
                ) / f_var.sum(dim=(1, 2)).reshape(-1, 1, 1)
                # Laplace Bridge
                _, K = f_mu.size(0), f_mu.size(-1)
                f_var_diag = torch.diagonal(f_var, dim1=1, dim2=2)
                # optional: variance correction
                if link_approx == "bridge_norm":
                    f_var_diag_mean = f_var_diag.mean(dim=1)
                    f_var_diag_mean /= torch.as_tensor(
                        [K / 2], device=self._device
                    ).sqrt()
                    f_mu /= f_var_diag_mean.sqrt().unsqueeze(-1)
                    f_var_diag /= f_var_diag_mean.unsqueeze(-1)
                sum_exp = torch.exp(-f_mu).sum(dim=1).unsqueeze(-1)
                alpha = (1 - 2 / K + f_mu.exp() / K**2 * sum_exp) / f_var_diag
                return torch.nan_to_num(alpha / alpha.sum(dim=1).unsqueeze(-1), nan=1.0)
        else:
            param_samples, samples = self._nn_predictive_samples(x, n_samples)
            if self.likelihood == "regression":
                return param_samples, samples.mean(dim=0), samples.var(dim=0)
            return param_samples, samples.mean(dim=0)
