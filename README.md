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

### Log-prior and log-likelihood

```python
from numpyro_inferutils import build_logprob_functions

logprior, loglik = build_logprob_functions(model)

theta = {
    "x": 0.0,
    "y": 1.2,
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
)
```

This inspects the model’s sample-site supports and applies the appropriate
inverse transforms using

```python
biject_to(site["fn"].support)
```

---

### Seeding and substituting parameters

```python
from jax import random
from numpyro_inferutils.transforms import seed_and_substitute

rng_key = random.PRNGKey(0)

model_sub = seed_and_substitute(
    model,
    params_dict={"sigma": 0.5},
    param_space="unconstrained",
    rng_key=rng_key,
)
```

- If `param_space="unconstrained"`, parameters are interpreted as living in
  unconstrained space and mapped to constrained space using NumPyro’s internal
  unconstraining reparameterization.
- If `param_space="constrained"`, values are substituted directly.

---

### Fisher information (independent Gaussian likelihood)

```python
from numpyro_inferutils.fisher import information_from_model_independent_normal

info = information_from_model_independent_normal(
    model=model,
    pdic={"w": 1.0, "b": 0.0},
    mu_name="mu",
    observed=y_obs,
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
