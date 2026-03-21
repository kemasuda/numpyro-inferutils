import numpy as np
import pytest
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from numpyro_inferutils.experimental.multinest import MultiNestRunner
import numpyro_inferutils.experimental.multinest as multinest_module


def linear_model(x, y, sigma=1.0):
    w = numpyro.sample("w", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    mu = w * x + b
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def gamma_model():
    lam = numpyro.sample("lam", dist.Gamma(2.0, 3.0))
    numpyro.sample("obs", dist.Normal(lam, 1.0), obs=jnp.array(0.0))


def dirichlet_model():
    p = numpyro.sample("p", dist.Dirichlet(jnp.ones(3)))
    numpyro.sample("obs", dist.Normal(p[0], 1.0), obs=jnp.array(0.0))


def test_multinest_runner_with_model_args_and_kwargs():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([0.5, 1.0, 1.5])

    runner = MultiNestRunner(
        linear_model,
        model_args=(x, y),
        model_kwargs={"sigma": 2.0},
    )

    assert runner.site_names == ["w", "b"]
    assert runner.ndim == 2

    theta = jnp.array([0.5, 0.1])
    ll = runner.loglikelihood(theta)

    expected = dist.Normal(0.5 * x + 0.1, 2.0).log_prob(y).sum()
    assert jnp.allclose(ll, expected)


def test_prior_returns_flat_vector_in_site_order():
    x = jnp.array([1.0, 2.0])
    y = jnp.array([0.0, 0.0])

    runner = MultiNestRunner(linear_model, model_args=(x, y))
    theta = runner.prior(jnp.array([0.5, 0.5]))

    assert isinstance(theta, np.ndarray)
    assert theta.shape == (2,)
    assert np.isfinite(theta).all()


def test_gamma_prior_transform_is_supported():
    runner = MultiNestRunner(gamma_model)

    theta = runner.prior(jnp.array([0.4]))
    assert theta.shape == (1,)
    assert np.isfinite(theta).all()
    assert theta[0] > 0.0


def test_site_overrides_are_used():
    x = jnp.array([1.0])
    y = jnp.array([0.0])

    runner = MultiNestRunner(
        linear_model,
        model_args=(x, y),
        site_overrides={
            "w": lambda q: jnp.array(10.0),
            "b": lambda q: jnp.array(-2.0),
        },
    )

    theta = runner.prior(jnp.array([0.1, 0.9]))
    assert np.allclose(theta, np.array([10.0, -2.0]))


def test_unsupported_prior_raises():
    with pytest.raises(NotImplementedError, match="Dirichlet"):
        MultiNestRunner(dirichlet_model)


def test_logposterior_matches_logprior_plus_loglik():
    x = jnp.array([1.0, 2.0])
    y = jnp.array([0.3, 0.7])

    runner = MultiNestRunner(linear_model, model_args=(x, y))
    theta = jnp.array([0.2, -0.1])

    lp = runner.logposterior(theta)
    expected = runner.logprior({"w": theta[0], "b": theta[1]}) + runner.loglik(
        {"w": theta[0], "b": theta[1]}
    )
    assert jnp.allclose(lp, expected)


def test_run_calls_solve(monkeypatch):
    called = {}

    class DummyAnalyzer:
        def __init__(self, n_params, outputfiles_basename):
            self.n_params = n_params
            self.outputfiles_basename = outputfiles_basename

    def fake_solve(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    def fake_import():
        return fake_solve, DummyAnalyzer

    monkeypatch.setattr(multinest_module, "_import_pymultinest", fake_import)

    x = jnp.array([1.0, 2.0])
    y = jnp.array([0.0, 0.0])

    runner = MultiNestRunner(linear_model, model_args=(x, y))
    result = runner.run(outputfiles_basename="chains/test-")

    assert result == {"ok": True}
    assert called["n_dims"] == 2
    assert called["outputfiles_basename"] == "chains/test-"
    assert callable(called["Prior"])
    assert callable(called["LogLikelihood"])


def test_get_analyzer_requires_run():
    x = jnp.array([1.0])
    y = jnp.array([0.0])

    runner = MultiNestRunner(linear_model, model_args=(x, y))
    with pytest.raises(RuntimeError, match="run\\(\\) must be called"):
        runner.get_analyzer()
