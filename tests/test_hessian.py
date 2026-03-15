import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from numpyro_inferutils.fisher import (
    information_from_model_independent_normal,
    hessian_from_model,
)


def linear_normal_model(x, y, sigma):
    a = numpyro.sample("a", dist.Normal(0.0, 10.0))
    b = numpyro.sample("b", dist.Normal(0.0, 10.0))
    mu = a * x + b
    numpyro.deterministic("mu", mu)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def test_hessian_loglik_matches_fisher_for_linear_gaussian():
    x = jnp.array([-2.0, -0.5, 0.3, 1.2, 2.0])
    sigma = jnp.ones_like(x) * 0.7
    y = jnp.array([1.1, 0.2, -0.3, 0.7, 1.9])

    pdic = {"a": jnp.array(0.4), "b": jnp.array(-0.2)}
    keys = ["a", "b"]

    info = information_from_model_independent_normal(
        model=linear_normal_model,
        model_args=(x, y, sigma),
        pdic=pdic,
        mu_name="mu",
        observed=y,
        keys=keys,
        sigma_sd=sigma,
        param_space="constrained",
        diff_mode="rev",
    )
    F = info["fisher"]

    res = hessian_from_model(
        model=linear_normal_model,
        model_args=(x, y, sigma),
        pdic=pdic,
        keys=keys,
        which="loglik",
        param_space="constrained",
        diff_mode="fwdrev",
    )
    H = res["hessian"]

    assert H.shape == (2, 2)
    assert jnp.allclose(-H, F, rtol=1e-6, atol=1e-6)


def test_hessian_loglik_matches_analytic_for_linear_gaussian():
    x = jnp.array([-2.0, -0.5, 0.3, 1.2, 2.0])
    sigma0 = 0.7
    sigma = jnp.ones_like(x) * sigma0
    y = jnp.array([1.1, 0.2, -0.3, 0.7, 1.9])

    pdic = {"a": jnp.array(0.4), "b": jnp.array(-0.2)}

    res = hessian_from_model(
        model=linear_normal_model,
        model_args=(x, y, sigma),
        pdic=pdic,
        keys=["a", "b"],
        which="loglik",
        param_space="constrained",
        diff_mode="fwdrev",
    )
    H = res["hessian"]

    H_expected = -(1.0 / sigma0**2) * jnp.array([
        [jnp.sum(x**2), jnp.sum(x)],
        [jnp.sum(x),    x.size],
    ])

    assert jnp.allclose(H, H_expected, rtol=1e-6, atol=1e-6)


def test_hessian_logprob_equals_loglik_plus_logprior():
    x = jnp.array([-2.0, -0.5, 0.3, 1.2, 2.0])
    sigma = jnp.ones_like(x) * 0.7
    y = jnp.array([1.1, 0.2, -0.3, 0.7, 1.9])

    pdic = {"a": jnp.array(0.4), "b": jnp.array(-0.2)}
    keys = ["a", "b"]

    Hlik = hessian_from_model(
        model=linear_normal_model,
        model_args=(x, y, sigma),
        pdic=pdic,
        keys=keys,
        which="loglik",
        param_space="constrained",
        diff_mode="fwdrev",
    )["hessian"]

    Hprior = hessian_from_model(
        model=linear_normal_model,
        model_args=(x, y, sigma),
        pdic=pdic,
        keys=keys,
        which="logprior",
        param_space="constrained",
        diff_mode="fwdrev",
    )["hessian"]

    Hprob = hessian_from_model(
        model=linear_normal_model,
        model_args=(x, y, sigma),
        pdic=pdic,
        keys=keys,
        which="logprob",
        param_space="constrained",
        diff_mode="fwdrev",
    )["hessian"]

    assert jnp.allclose(Hprob, Hlik + Hprior, rtol=1e-6, atol=1e-6)
