# tests/test_transforms.py

import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.distributions.transforms import biject_to

from numpyro_inferutils.transforms import (
    to_unconstrained_dict,
    seed_and_substitute,
)


# ----------------------------------------------------------------------
# Helper models
# ----------------------------------------------------------------------

def positive_scale_model():
    """
    Simple model with a strictly positive parameter sigma:

        sigma ~ LogNormal(0, 1)
        y ~ Normal(0, sigma)  (observed)

    This is used to test constrained/unconstrained conversions.
    """
    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 1.0))
    numpyro.sample("y", dist.Normal(0.0, sigma), obs=jnp.array(0.0))


def normal_location_model():
    """
    Simple model with an unconstrained parameter x:

        x ~ Normal(0, 1)
        y ~ Normal(x, 1)  (observed)

    Used to test seed_and_substitute with param_space="constrained".
    """
    x = numpyro.sample("x", dist.Normal(0.0, 1.0))
    numpyro.sample("y", dist.Normal(x, 1.0), obs=jnp.array(0.0))


# ----------------------------------------------------------------------
# Tests for to_unconstrained_dict
# ----------------------------------------------------------------------

def test_to_unconstrained_dict_roundtrip_positive_scale():
    """
    For a positive parameter sigma, check that transforming a constrained
    value to unconstrained space via to_unconstrained_dict and then applying
    the model's bijector recovers the original constrained value.
    """
    sigma_constrained = jnp.array(2.0)
    params_constrained = {"sigma": sigma_constrained}
    keys = ["sigma"]

    z_dict = to_unconstrained_dict(
        positive_scale_model, params_constrained, keys
    )
    z = z_dict["sigma"]

    # Build the same bijector that the model uses for sigma
    # (LogNormal(0, 1) -> support is positive reals)
    d = dist.LogNormal(0.0, 1.0)
    support = d.support
    bij = biject_to(support)

    sigma_recovered = bij(z)

    assert jnp.allclose(sigma_recovered, sigma_constrained)


# ----------------------------------------------------------------------
# Tests for seed_and_substitute (constrained)
# ----------------------------------------------------------------------

def test_seed_and_substitute_constrained_overrides_value():
    """
    When param_space='constrained', seed_and_substitute should directly
    substitute the provided constrained values into the model.
    """
    rng_key = random.PRNGKey(1)

    model_seeded = seed_and_substitute(
        normal_location_model,
        params_dict={"x": jnp.array(1.23)},
        param_space="constrained",
        rng_key=rng_key,
    )

    tr = handlers.trace(model_seeded).get_trace()
    assert "x" in tr
    assert jnp.allclose(tr["x"]["value"], 1.23)


# ----------------------------------------------------------------------
# Tests for seed_and_substitute (unconstrained)
# ----------------------------------------------------------------------

def test_seed_and_substitute_unconstrained_matches_bijector():
    """
    For a positive parameter sigma, when param_space='unconstrained',
    the value provided in params_dict should be interpreted as an
    unconstrained parameter and mapped to the constrained space
    via the same bijector as biject_to(site['fn'].support).
    """
    rng_key = random.PRNGKey(2)

    # Construct an explicit bijector matching positive_scale_model's sigma
    d = dist.LogNormal(0.0, 1.0)
    support = d.support
    bij = biject_to(support)

    # Choose an unconstrained value z and its constrained counterpart sigma_expected
    z = jnp.array(0.5)
    sigma_expected = bij(z)

    model_seeded = seed_and_substitute(
        positive_scale_model,
        params_dict={"sigma": z},          # unconstrained value
        param_space="unconstrained",
        rng_key=rng_key,
    )

    tr = handlers.trace(model_seeded).get_trace()
    assert "sigma" in tr
    sigma_value = tr["sigma"]["value"]

    # The constrained value in the trace should match bij(z)
    assert jnp.allclose(sigma_value, sigma_expected)


def test_seed_and_substitute_unconstrained_roundtrip_with_to_unconstrained():
    """
    Round-trip test:

        constrained -> (to_unconstrained_dict) -> unconstrained z
                    -> (seed_and_substitute, param_space='unconstrained')
                    -> constrained

    The final constrained value in the model trace should match the original
    constrained parameter (up to numerical tolerance).
    """
    rng_key = random.PRNGKey(3)

    sigma_original = jnp.array(1.7)
    params_constrained = {"sigma": sigma_original}
    keys = ["sigma"]

    # constrained -> unconstrained
    z_dict = to_unconstrained_dict(
        positive_scale_model, params_constrained, keys
    )
    z = z_dict["sigma"]

    # unconstrained -> constrained via seed_and_substitute
    model_seeded = seed_and_substitute(
        positive_scale_model,
        params_dict={"sigma": z},
        param_space="unconstrained",
        rng_key=rng_key,
    )

    tr = handlers.trace(model_seeded).get_trace()
    sigma_final = tr["sigma"]["value"]

    assert jnp.allclose(sigma_final, sigma_original, rtol=1e-6, atol=1e-7)
