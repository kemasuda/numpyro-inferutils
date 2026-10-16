import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro import handlers

from numpyro_inferutils.fisher import (
    hessian_from_model,
    information_from_model_independent_normal,
)
from numpyro_inferutils.transforms import (
    seed_and_substitute,
    to_unconstrained_dict,
)


def dependent_support_model(width, *, y):
    lower = numpyro.sample("lower", dist.Uniform(0.0, 1.0))
    lower_bound = numpyro.deterministic("lower_bound", lower)
    x = numpyro.sample("x", dist.Uniform(lower_bound, lower_bound + width))
    mean = numpyro.deterministic("mean", jnp.array([x]))
    numpyro.sample("obs", dist.Normal(mean, 1.0), obs=y)


@pytest.mark.parametrize("keys", [("lower", "x"), ("x",)])
def test_to_unconstrained_uses_supplied_dependent_support(keys):
    # Extra deterministic/observed entries must not override the model.
    params = {"lower": 0.25, "x": 0.75, "lower_bound": -10.0, "obs": -100.0}
    y = jnp.array([0.75])
    result = to_unconstrained_dict(
        dependent_support_model, params, keys, 1.0, y=y,
    )

    assert set(result) == set(keys)
    assert jnp.allclose(result["x"], 0.0, atol=1e-7)
    if "lower" in keys:
        assert jnp.allclose(result["lower"], jnp.log(1.0 / 3.0))

    z = {"lower": jnp.log(1.0 / 3.0), **result}
    traced = handlers.trace(seed_and_substitute(
        dependent_support_model, z, "unconstrained", random.PRNGKey(1),
    )).get_trace(1.0, y=y)
    assert jnp.allclose(traced["lower"]["value"], params["lower"])
    assert jnp.allclose(traced["x"]["value"], params["x"])


@pytest.mark.parametrize("diff_mode", ["fwd", "rev"])
@pytest.mark.parametrize("param_space", ["constrained", "unconstrained"])
def test_information_with_parameter_dependent_support(diff_mode, param_space):
    # Regression from jnkepler: x = sigmoid(z_lower) + sigmoid(z_x).
    result = information_from_model_independent_normal(
        model=dependent_support_model,
        model_args=(1.0,),
        model_kwargs={"y": jnp.array([0.75])},
        pdic={"lower": 0.25, "x": 0.75, "lower_bound": -10.0},
        keys=["lower", "x"],
        mu_name="mean",
        obs_name="obs",
        sigma_sd=jnp.ones(1),
        param_space=param_space,
        diff_mode=diff_mode,
    )

    if param_space == "unconstrained":
        assert jnp.allclose(result["params_unconstrained"]["lower"], jnp.log(1.0 / 3.0))
        assert jnp.allclose(result["params_unconstrained"]["x"], 0.0, atol=1e-7)
        derivative = jnp.array([0.25 * 0.75, 0.5 * 0.5])
    else:
        derivative = jnp.array([0.0, 1.0])
    expected = jnp.outer(derivative, derivative)
    assert jnp.all(jnp.isfinite(result["fisher"]))
    assert jnp.allclose(result["fisher"], expected, rtol=1e-6, atol=1e-7)
    assert set(result["params_unconstrained"]) == {"lower", "x"}


@pytest.mark.parametrize("diff_mode", ["fwdrev", "revfwd", "revrev", "fwdfwd"])
def test_hessian_with_parameter_dependent_support(diff_mode):
    result = hessian_from_model(
        model=dependent_support_model,
        model_args=(1.0,),
        model_kwargs={"y": jnp.array([0.75])},
        pdic={"lower": 0.25, "x": 0.75, "lower_bound": -10.0},
        keys=["lower", "x"],
        which="logprob",
        param_space="unconstrained",
        diff_mode=diff_mode,
    )

    # The priors are uniform. In unconstrained space the log density also
    # includes the two log-Jacobians of the sigmoid transforms.
    derivative = jnp.array([0.25 * 0.75, 0.5 * 0.5])
    expected = -jnp.outer(derivative, derivative) - 2.0 * jnp.diag(derivative)
    assert jnp.allclose(result["hessian"], expected, rtol=1e-6, atol=1e-7)
    expected_value = -0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(derivative).sum()
    assert jnp.allclose(result["value"], expected_value)


@pytest.mark.parametrize("param_space", ["constrained", "unconstrained"])
@pytest.mark.parametrize("kind", ["fisher", "hessian"])
def test_parameter_discovery_uses_valid_supplied_values(param_space, kind):
    def model():
        # The broad prior allows values outside the likelihood's valid domain.
        variance = numpyro.sample("variance", dist.Normal(0.0, 1.0))
        scale = numpyro.deterministic("scale", jnp.sqrt(variance))
        x = numpyro.sample("x", dist.Normal(0.0, 1.0))
        mean = numpyro.deterministic("mean", jnp.array([x]))
        numpyro.sample(
            "obs", dist.Normal(mean, scale, validate_args=True),
            obs=jnp.array([0.75]),
        )

    kwargs = dict(
        model=model,
        pdic={"variance": 4.0, "x": 0.75, "scale": -1.0, "obs": -100.0},
        keys=["x"],
        param_space=param_space,
    )
    if kind == "fisher":
        result = information_from_model_independent_normal(
            **kwargs, mu_name="mean", obs_name="obs", sigma_sd=jnp.array([2.0]),
        )
        assert jnp.allclose(result["fisher"], jnp.array([[0.25]]))
    else:
        result = hessian_from_model(**kwargs, which="loglik")
        assert jnp.allclose(result["hessian"], jnp.array([[-0.25]]))
        expected_value = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(2.0)
        assert jnp.allclose(result["value"], expected_value)
    assert set(result["params_unconstrained"]) == {"variance", "x"}
