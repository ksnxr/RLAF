import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_ivp, solve_bvp
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController, ConstantStepSize


def get_np_monge_gaussian_fun(dim, precision):
    def func(t, y):
        theta = y[:dim, :]
        v = y[dim:, :]
        theta_grad = -precision @ theta
        norm_theta_grad_2 = np.sum(np.square(theta_grad), axis=0)

        W_2 = 1.0 + norm_theta_grad_2
        mho = -np.sum(v * (precision @ v), axis=0) / W_2

        return np.concatenate([v, -mho * theta_grad])

    return func


def get_np_velocity(dim, precision, v):
    # https://acme.byu.edu/0000017c-ccfe-da17-a5fd-cdfeff540000/acmefiles-08-bvp-2021-pdf
    fun = get_np_monge_gaussian_fun(dim, precision)

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


def get_np_christoffel_fun(dim, christoffel_fn):
    def func(t, y):
        theta = y[:dim]
        v = y[dim:]
        a = -christoffel_fn(theta, v)

        return np.concatenate([v, a])

    return func


def get_jax_christoffel_fun(dim, christoffel_fn):
    def func(t, y, args):
        theta = y[:dim]
        v = y[dim:]
        a = -christoffel_fn(theta, v)

        return jnp.concatenate([v, a])

    return func


def np_christoffel_geodesic(dim, christoffel_fn, theta, v):
    fun = get_np_christoffel_fun(dim, christoffel_fn=christoffel_fn)
    return solve_ivp(
        fun=fun,
        t_span=(0.0, 1.0),
        y0=np.concatenate([theta, v]),
    )


def jax_christoffel_geodesic(dim, christoffel_fn, theta, v):
    fun = get_jax_christoffel_fun(dim, christoffel_fn=christoffel_fn)
    term = ODETerm(fun)
    solver = Dopri5()
    y0 = jnp.concatenate([theta, v])
    return diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=None,
        y0=y0,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-6),
        throw=False,
    )


def jax_christoffel_geodesic_lines(dim, christoffel_fn, theta, v, saveat):
    fun = get_jax_christoffel_fun(dim, christoffel_fn=christoffel_fn)
    term = ODETerm(fun)
    solver = Dopri5()
    y0 = jnp.concatenate([theta, v])
    return diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=None,
        y0=y0,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-6),
        saveat=saveat,
    )


def jax_christoffel_geodesic_const_stepsize(dim, christoffel_fn, theta, v):
    fun = get_jax_christoffel_fun(dim, christoffel_fn=christoffel_fn)
    term = ODETerm(fun)
    solver = Dopri5()
    y0 = jnp.concatenate([theta, v])
    return diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=2e-4,
        max_steps=6000,
        y0=y0,
        stepsize_controller=ConstantStepSize(),
    )
