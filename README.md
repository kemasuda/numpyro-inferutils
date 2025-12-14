# numpyro-inferutils

Small utility functions for inference with NumPyro models.

This package provides lightweight helpers for:
- extracting log-prior and log-likelihood from NumPyro models,
- working with constrained / unconstrained parameter spaces,
- computing Fisher information matrices from NumPyro models with
  independent Gaussian likelihoods.

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

logprior, loglik = build_logprob_functions(model, model_kwargs={"x": x, "y": y})

theta = {
    "w": 0.0,
    "b": 1.2,
}

lp = logprior(theta)
ll = loglik(theta)
```

- `logprior(theta)` sums log-probabilities from *non-observed* sample sites.
- `loglik(theta)` sums log-probabilities from *observed* sample sites.
- Contributions added via `numpyro.factor` are treated as part of the
  log-likelihood.

---

### Constrained ↔ unconstrained parameters

```python
from numpyro_inferutils.transforms import to_unconstrained_dict

params_constrained = {"sigma": 2.0}
params_unconstrained = to_unconstrained_dict(
    model,
    params_constrained,
    keys=["sigma"],
    x=x, y=y
)
```

This inspects the model’s sample-site supports and applies the appropriate
inverse transforms using

```python
biject_to(site["fn"].support)
```

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

The Fisher matrix is approximated as

F ≈ Jᵀ J,

where J_ij = ∂r_i / ∂θ_j and

r = (y − μ(θ)) / σ.

Both constrained and unconstrained parameterizations are supported.

---

## License

MIT License.
