import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from numpyro_inferutils import find_map_svi


def model_mvn(y, cov, mu0, cov0):
    theta = numpyro.sample(
        "theta",
        dist.MultivariateNormal(mu0, covariance_matrix=cov0),
    )
    numpyro.sample(
        "y",
        dist.MultivariateNormal(theta, covariance_matrix=cov),
        obs=y,
    )


def test_find_map_svi_multivariate_normal():
    key = jax.random.PRNGKey(0)

    dim = 3

    mu0 = jnp.array([0.5, -0.3, 0.1])
    cov0 = jnp.array([
        [1.0, 0.2, 0.0],
        [0.2, 1.5, 0.1],
        [0.0, 0.1, 0.8],
    ])

    cov = jnp.array([
        [0.4, 0.1, 0.0],
        [0.1, 0.3, 0.05],
        [0.0, 0.05, 0.2],
    ])

    # generate synthetic observation
    true_theta = jnp.array([0.8, -0.1, 0.2])
    y = true_theta + jax.random.multivariate_normal(
        key, jnp.zeros(dim), cov
    )

    # analytic MAP
    cov0_inv = jnp.linalg.inv(cov0)
    cov_inv = jnp.linalg.inv(cov)
    cov_map = jnp.linalg.inv(cov0_inv + cov_inv)
    theta_map = cov_map @ (cov0_inv @ mu0 + cov_inv @ y)

    p_fit = find_map_svi(
        lambda: model_mvn(y=y, cov=cov, mu0=mu0, cov0=cov0),
        rng_key=jax.random.PRNGKey(1),
        step_size=5e-2,
        num_steps=3000,
        progress_bar=False,
    )

    assert "theta" in p_fit
    assert p_fit["theta"].shape == (dim,)
    assert jnp.all(jnp.isfinite(p_fit["theta"]))

    assert jnp.allclose(p_fit["theta"], theta_map, atol=3e-2)
    print(p_fit["theta"], theta_map)


if __name__ == '__main__':
    test_find_map_svi_multivariate_normal()
