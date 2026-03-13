# tests/test_fisher.py

import jax.numpy as jnp
import pytest
from jax import random, jacrev
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import biject_to

from numpyro_inferutils.fisher import information_from_model_independent_normal


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------

def linear_gaussian_model(x, sigma):
    """
    y_i = w * x_i + b + eps_i,  eps_i ~ N(0, sigma)

    Parameters:
        w, b ~ Normal(0, 1)

    Deterministic site:
        mu = w * x + b

    Observed site:
        obs ~ Normal(mu, sigma)
    """
    w = numpyro.sample("w", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    mu = w * x + b
    numpyro.deterministic("mu", mu)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=None)


def analytic_fisher_linear(x, sigma):
    """
    Analytic Fisher information matrix for (w, b) in the linear Gaussian model:

        y_i = w * x_i + b + eps_i,  eps_i ~ N(0, sigma)

    For iid data, the Fisher information in (w, b) is

        F = (1/sigma^2) * [[sum x_i^2, sum x_i],
                           [sum x_i,   N      ]]
    """
    N = x.size
    s1 = jnp.sum(x)
    s2 = jnp.sum(x**2)
    factor = 1.0 / (sigma**2)
    return factor * jnp.array([[s2, s1],
                               [s1, N]], dtype=x.dtype)


def positive_param_model(x, sigma):
    """
    Toy model with a strictly positive parameter alpha:

        y_i = alpha * x_i + eps_i,  eps_i ~ N(0, sigma)

    - parameter: alpha > 0 (LogNormal prior just to enforce support)
    - deterministic: mu = alpha * x
    - observed site: obs ~ N(mu, sigma)
    """
    alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 1.0))
    mu = alpha * x
    numpyro.deterministic("mu", mu)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=None)


def two_block_linear_gaussian_model(x1, x2, sigma1, sigma2, y1=None, y2=None):
    """Two independent Gaussian blocks sharing the same linear mean parameters."""
    w = numpyro.sample("w", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))

    mu1 = w * x1 + b
    mu2 = w * x2 + b

    numpyro.deterministic("mu1", mu1)
    numpyro.deterministic("mu2", mu2)

    numpyro.sample("obs1", dist.Normal(mu1, sigma1), obs=y1)
    numpyro.sample("obs2", dist.Normal(mu2, sigma2), obs=y2)


def analytic_fisher_two_block_linear(x1, sigma1, x2, sigma2):
    """Analytic Fisher for two independent linear-Gaussian blocks."""
    return analytic_fisher_linear(x1, sigma1) + analytic_fisher_linear(x2, sigma2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_information_from_model_independent_normal_linear():
    """
    Check that for a simple linear-Gaussian model with parameters (w, b),
    the Fisher matrix returned by information_from_model_independent_normal
    matches the analytic Fisher.
    """
    key = random.PRNGKey(0)
    x = jnp.linspace(-1.0, 1.0, 5)
    sigma = 0.3
    sigma_sd = jnp.full_like(x, sigma)

    # choose some parameters and corresponding noiseless "observations"
    w0 = 1.5
    b0 = -0.2
    mu0 = w0 * x + b0
    y_obs = mu0

    pdic = {"w": jnp.array(w0), "b": jnp.array(b0)}

    info = information_from_model_independent_normal(
        model=linear_gaussian_model,
        model_args=(x, sigma),
        model_kwargs=None,
        pdic=pdic,
        mu_name="mu",
        observed=y_obs,
        obs_name="obs",
        keys=["w", "b"],
        sigma_sd=sigma_sd,
        param_space="unconstrained",
        rng_key=key,
        diff_mode="rev",
    )

    F = info["fisher"]
    F_expected = analytic_fisher_linear(x, sigma)

    assert F.shape == (2, 2)
    assert jnp.allclose(F, F_expected, rtol=1e-4, atol=1e-6)


def test_information_from_model_independent_normal_constrained_param_space():
    """
    For parameters with full real support (Normal prior), the Fisher matrix
    should be the same whether we treat the parameters as living in the
    constrained or unconstrained space.
    """
    key = random.PRNGKey(1)
    x = jnp.linspace(0.0, 1.0, 4)
    sigma = 0.5
    sigma_sd = jnp.full_like(x, sigma)

    w0 = 0.7
    b0 = 0.1
    mu0 = w0 * x + b0
    y_obs = mu0

    pdic = {"w": jnp.array(w0), "b": jnp.array(b0)}

    info_c = information_from_model_independent_normal(
        model=linear_gaussian_model,
        model_args=(x, sigma),
        pdic=pdic,
        mu_name="mu",
        observed=y_obs,
        obs_name="obs",
        keys=["w", "b"],
        sigma_sd=sigma_sd,
        param_space="constrained",
        rng_key=key,
        diff_mode="rev",
    )

    info_u = information_from_model_independent_normal(
        model=linear_gaussian_model,
        model_args=(x, sigma),
        pdic=pdic,
        mu_name="mu",
        observed=y_obs,
        obs_name="obs",
        keys=["w", "b"],
        sigma_sd=sigma_sd,
        param_space="unconstrained",
        rng_key=key,
        diff_mode="rev",
    )

    F_c = info_c["fisher"]
    F_u = info_u["fisher"]

    assert F_c.shape == (2, 2)
    assert F_u.shape == (2, 2)
    assert jnp.allclose(F_c, F_u, rtol=1e-5, atol=1e-7)


def test_information_unconstrained_positive_param_matches_manual_chainrule():
    """
    Explicitly test the accuracy of the *unconstrained* Fisher.

    We use a positive parameter alpha with a LogNormal prior (support (0, inf)),
    and define an explicit bijector z -> alpha. We then:

        1) compute Fisher in z-space using information_from_model_independent_normal
           with param_space='unconstrained'
        2) compute Fisher in z-space manually via J^T J where
           J = d r(z) / d z, r = standardized residuals

    These two should agree up to numerical tolerance if the unconstrained
    handling is correct.
    """
    key = random.PRNGKey(0)

    # Data setup
    x = jnp.linspace(0.1, 1.0, 7)  # avoid all zeros
    sigma = 0.3
    sigma_sd = jnp.full_like(x, sigma)

    alpha0 = 1.2  # constrained > 0
    y_obs = alpha0 * x  # noiseless observations

    # --- 1) Fisher from our helper, in UNCONSTRAINED space ---

    pdic_constrained = {"alpha": jnp.array(alpha0)}

    info = information_from_model_independent_normal(
        model=positive_param_model,
        model_args=(x, sigma),
        model_kwargs=None,
        pdic=pdic_constrained,
        mu_name="mu",
        observed=y_obs,
        obs_name="obs",
        keys=["alpha"],
        sigma_sd=sigma_sd,
        param_space="unconstrained",  # unconstrained Fisher we want to test
        rng_key=key,
        diff_mode="rev",
    )

    F_unconstrained = info["fisher"]
    assert F_unconstrained.shape == (1, 1)

    # --- 2) Reference Fisher in z (unconstrained) via explicit chain rule ---

    # Use the same support that the model uses for alpha
    d = dist.LogNormal(0.0, 1.0)
    support = d.support
    bij = biject_to(support)  # z -> alpha (e.g., exp)

    # unconstrained value z0 corresponding to alpha0
    z0 = bij.inv(jnp.array(alpha0))

    def residuals_z(z):
        alpha = bij(z)
        mu = alpha * x
        r = (y_obs - mu) / sigma  # standardized residuals
        return r  # shape (N,)

    # Jacobian wrt z (unconstrained)
    J = jacrev(residuals_z)(z0).reshape(x.size, 1)
    F_ref = J.T @ J  # (1,1)

    # --- 3) Compare ---

    assert F_ref.shape == (1, 1)
    assert jnp.allclose(F_unconstrained, F_ref, rtol=1e-5, atol=1e-7)


def test_information_from_model_independent_normal_multiple_mu_block_inputs():
    """Multiple deterministic sites can be concatenated blockwise."""
    key = random.PRNGKey(2)

    x1 = jnp.array([-1.0, 0.5, 1.5])
    x2 = jnp.array([0.2, 1.2])
    sigma1 = 0.3
    sigma2 = 0.7

    w0 = 0.8
    b0 = -0.4
    y1 = w0 * x1 + b0
    y2 = w0 * x2 + b0

    info = information_from_model_independent_normal(
        model=two_block_linear_gaussian_model,
        model_args=(x1, x2, sigma1, sigma2),
        model_kwargs={"y1": y1, "y2": y2},
        pdic={"w": jnp.array(w0), "b": jnp.array(b0)},
        mu_name=["mu1", "mu2"],
        observed=[y1, y2],
        obs_name=["obs1", "obs2"],
        keys=["w", "b"],
        sigma_sd=[jnp.full_like(x1, sigma1), jnp.full_like(x2, sigma2)],
        param_space="unconstrained",
        rng_key=key,
        diff_mode="rev",
    )

    F = info["fisher"]
    F_expected = analytic_fisher_two_block_linear(x1, sigma1, x2, sigma2)

    assert F.shape == (2, 2)
    assert jnp.allclose(F, F_expected, rtol=1e-4, atol=1e-6)


def test_information_from_model_independent_normal_multiple_mu_obs_from_trace():
    """Multiple observed sites can also be read from the model trace."""
    key = random.PRNGKey(3)

    x1 = jnp.array([-0.5, 0.0, 1.0])
    x2 = jnp.array([0.3, 1.7])
    sigma1 = 0.4
    sigma2 = 0.6

    w0 = 1.1
    b0 = 0.2
    y1 = w0 * x1 + b0
    y2 = w0 * x2 + b0

    info = information_from_model_independent_normal(
        model=two_block_linear_gaussian_model,
        model_args=(x1, x2, sigma1, sigma2),
        model_kwargs={"y1": y1, "y2": y2},
        pdic={"w": jnp.array(w0), "b": jnp.array(b0)},
        mu_name=["mu1", "mu2"],
        obs_name=["obs1", "obs2"],
        keys=["w", "b"],
        sigma_sd=jnp.concatenate([jnp.full_like(x1, sigma1), jnp.full_like(x2, sigma2)]),
        param_space="unconstrained",
        rng_key=key,
        diff_mode="rev",
    )

    F = info["fisher"]
    F_expected = analytic_fisher_two_block_linear(x1, sigma1, x2, sigma2)

    assert F.shape == (2, 2)
    assert jnp.allclose(F, F_expected, rtol=1e-4, atol=1e-6)


def test_information_from_model_independent_normal_multiple_mu_shape_mismatch():
    """Blockwise sigma must match the corresponding deterministic block sizes."""
    key = random.PRNGKey(4)

    x1 = jnp.array([0.0, 1.0, 2.0])
    x2 = jnp.array([0.5, 1.5])
    sigma1 = 0.3
    sigma2 = 0.5

    w0 = 0.5
    b0 = 0.1
    y1 = w0 * x1 + b0
    y2 = w0 * x2 + b0

    with pytest.raises(ValueError, match=r"sigma_sd\[1\]"):
        information_from_model_independent_normal(
            model=two_block_linear_gaussian_model,
            model_args=(x1, x2, sigma1, sigma2),
            model_kwargs={"y1": y1, "y2": y2},
            pdic={"w": jnp.array(w0), "b": jnp.array(b0)},
            mu_name=["mu1", "mu2"],
            observed=[y1, y2],
            keys=["w", "b"],
            sigma_sd=[jnp.full_like(x1, sigma1), jnp.ones(3)],
            param_space="unconstrained",
            rng_key=key,
            diff_mode="rev",
        )
