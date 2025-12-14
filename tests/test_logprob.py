import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from numpyro_inferutils import build_logprob_functions


def simple_normal_model():
    x = numpyro.sample("x", dist.Normal(0.0, 1.0))
    numpyro.sample("y", dist.Normal(x, 1.0), obs=jnp.array(0.5))
    numpyro.factor("log_factor", -(x - 0.3)**2 / 0.1**2)


def test_build_logprob_functions_scalar():
    logprior, loglik = build_logprob_functions(simple_normal_model)

    theta = {"x": jnp.array(0.2)}

    lp = logprior(theta)
    ll = loglik(theta)

    # prior: Normal(0,1) at x=0
    expected_lp = dist.Normal(0.0, 1.0).log_prob(theta['x'])
    # likelihood: Normal(x,1) at y=0.5, x=0 + log_factor
    expected_ll = dist.Normal(theta['x'], 1.0).log_prob(
        0.5) - (theta['x'] - 0.3)**2 / 0.1**2

    assert jnp.allclose(lp, expected_lp)
    assert jnp.allclose(ll, expected_ll)


def vector_model():
    x = numpyro.sample("x", dist.Normal(jnp.zeros(3), jnp.ones(3)))
    numpyro.sample("y", dist.Normal(x, 1.0), obs=jnp.array([0.0, 1.0, 2.0]))


def test_build_logprob_functions_vector_sum():
    logprior, loglik = build_logprob_functions(vector_model)

    theta = {"x": jnp.array([0.0, 0.0, 0.0])}

    lp = logprior(theta)
    ll = loglik(theta)

    # prior: 3 independent N(0,1)
    expected_lp = dist.Normal(0.0, 1.0).log_prob(0.0) * 3
    # likelihood: sum of N(0,1) at y=[0,1,2]
    y = jnp.array([0.0, 1.0, 2.0])
    expected_ll = dist.Normal(0.0, 1.0).log_prob(y).sum()

    assert jnp.allclose(lp, expected_lp)
    assert jnp.allclose(ll, expected_ll)
