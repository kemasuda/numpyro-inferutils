# numpyro-inferutils

Small utility functions for inference with NumPyro models.

This package provides lightweight helpers for:
- extracting log-prior and log-likelihood from NumPyro models,
- working with constrained / unconstrained parameter spaces,
- computing Fisher information matrices from NumPyro models with independent Gaussian likelihoods,
- computing Hessian matrices of log-likelihood / log-prior / log-posterior directly from NumPyro models,
- performing MAP estimation using stochastic variational inference (SVI).

---

## Installation

```bash
pip install numpyro-inferutils
```

---

## Quick examples

### A minimal NumPyro model

All examples below assume a simple NumPyro model such as:

```python
import numpyro
import numpyro.distributions as dist
import numpy as np

x = np.linspace(-5, 5, 100)
sigma = np.ones_like(x) * np.exp(0.01)
y = 0.5 * x + 1.0 + np.random.randn(len(x)) * sigma

def model(x, y):
    w = numpyro.sample("w", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 0.01))
    mu = w * x + b
    numpyro.deterministic("mu", mu)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
```

### Log-prior and log-likelihood

```python
from numpyro_inferutils import build_logprob_functions

logprior, loglik = build_logprob_functions(model, model_args=(x, y))

theta = {
    "w": 0.0,
    "b": 1.2,
}

lp = logprior(theta)
ll = loglik(theta)
```

- `logprior(theta)` sums log-probabilities from *non-observed* sample sites.
- `loglik(theta)` sums log-probabilities from *observed* sample sites.
- Contributions added via `numpyro.factor` are treated as part of the log-likelihood.

---

### Constrained ↔ unconstrained parameters

```python
from numpyro_inferutils.transforms import to_unconstrained_dict

params_constrained = {"sigma": 2.0}
params_unconstrained = to_unconstrained_dict(
    model, params_constrained, keys=["sigma"], x=x, y=y
)
```

This inspects the model’s sample-site supports and applies the appropriate inverse transforms using

```python
biject_to(site["fn"].support)
```

---

### MAP estimation via SVI

For many applications, it is useful to obtain a fast maximum a posteriori (MAP) estimate, for example as an initial point for NUTS.

```python
import jax
from numpyro_inferutils import find_map_svi

rng_key = jax.random.PRNGKey(0)

p_map = find_map_svi(
    model,
    step_size=1e-2,
    num_steps=5_000,
    rng_key=rng_key,
    x=x,
    y=y,
)
```

- The MAP estimate is obtained via stochastic variational inference (SVI) using a Laplace autoguide (`AutoLaplaceApproximation`).
- Only a MAP-like point estimate (the guide median) is returned; the covariance of the Laplace approximation is intentionally not used.
- Parameter constraints defined in the NumPyro model are handled automatically.
- The returned parameters are in the constrained space.

---

### Fisher information (independent Gaussian likelihood)

```python
from numpyro_inferutils.fisher import information_from_model_independent_normal

info = information_from_model_independent_normal(
    model=model,
    pdic={"w": 1.0, "b": 0.5},
    mu_name="mu",
    observed=y,
    model_args=(x, y),
    keys=["w", "b"],
    sigma_sd=sigma,
)

F = info["fisher"]
```

The Fisher matrix for an independent Gaussian likelihood is computed as
F = Jᵀ J, where J_ij = ∂r_i / ∂θ_j and r = (y − μ(θ)) / σ.

Both constrained and unconstrained parameterizations are supported.
When the model mean is split across multiple deterministic sites, `mu_name` may also be given as a list or tuple.
In that case, the corresponding mean vectors are flattened and concatenated before constructing the standardized residuals.

The same convention is supported for `observed`, `obs_name`, and `sigma_sd`:
each may be passed either as one already-concatenated 1D array, or as a list/tuple matching the blocks in `mu_name`.

```python
info = information_from_model_independent_normal(
    model=model,
    pdic={"w": 1.0, "b": 0.5},
    mu_name=["mu_flux", "mu_rv"],
    observed=[y_flux, y_rv],
    sigma_sd=[sigma_flux, sigma_rv],
    model_args=(x_flux, x_rv, y_flux, y_rv),
    keys=["w", "b"],
)

F = info["fisher"]
```

The final concatenated shapes of `mu`, `observed`, and `sigma_sd` must agree.

---

### Hessian from a NumPyro model

```python
from numpyro_inferutils.fisher import hessian_from_model

res = hessian_from_model(
    model=model,
    model_args=(x, y),
    pdic={"w": 1.0, "b": 0.5},
    keys=["w", "b"],
    which="logprob",              # or "loglik", "logprior"
    param_space="unconstrained",  # or "constrained"
)

H = res["hessian"]
```

This function computes the Hessian of a scalar objective constructed directly from a NumPyro model.

- `which="loglik"` returns the Hessian of the log-likelihood.
- `which="logprior"` returns the Hessian of the log-prior.
- `which="logprob"` returns the Hessian of the full log-posterior up to an additive constant.

The returned matrix follows the parameter order specified by `keys`.
As in the Fisher helper, array-valued parameters are flattened and concatenated in a stable order, and the result dictionary includes `col_names` and `col_slices`.

```python
H = res["hessian"]
col_names = res["col_names"]
```

If you need the curvature of the negative log-posterior or the observed information matrix, use `-H`.

For an independent Gaussian likelihood with fixed standard deviations and a model mean that is linear in the parameters, -H` for `which="loglik" agrees with the Fisher matrix returned by
`information_from_model_independent_normal(...)`.

---

## License

MIT License.
