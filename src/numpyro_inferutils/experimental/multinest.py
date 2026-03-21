__all__ = ["MultiNestRunner"]

import numpy as np
import jax.numpy as jnp
from jax import random
from numpyro.handlers import seed, trace
import numpyro.distributions as dist

from ..logprob import build_logprob_functions

try:
    import arviz as az
except ImportError:
    az = None


def _import_pymultinest():
    try:
        from pymultinest.solve import solve
        from pymultinest.analyse import Analyzer
    except Exception as e:
        raise ImportError(
            "PyMultiNest is required for MultiNestRunner. "
            "Install pymultinest and the MultiNest library."
        ) from e
    return solve, Analyzer


def _safe_unit_interval(u):
    u = jnp.asarray(u)
    dtype = jnp.result_type(u, float)
    tiny = jnp.finfo(dtype).tiny
    eps = jnp.finfo(dtype).eps
    return jnp.clip(u, tiny, 1.0 - eps)


def _site_shape_and_size(value):
    shape = tuple(jnp.shape(value))
    size = int(np.prod(shape)) if len(shape) > 0 else 1
    return shape, size


def _flatten_theta_dict(theta_dict, site_info):
    parts = [jnp.ravel(jnp.asarray(theta_dict[s["name"]])) for s in site_info]
    if not parts:
        return jnp.zeros((0,), dtype=float)
    return jnp.concatenate(parts)


def _unflatten_theta_vector(theta, site_info):
    theta = jnp.asarray(theta)
    out = {}
    i = 0
    for s in site_info:
        n = s["size"]
        out[s["name"]] = theta[i:i + n].reshape(s["shape"])
        i += n
    return out


def _split_posterior_array(arr, site_info):
    arr = np.asarray(arr)
    out = {}
    i = 0
    for s in site_info:
        n = s["size"]
        out[s["name"]] = arr[:, i:i + n].reshape((arr.shape[0],) + s["shape"])
        i += n
    return out


def _uniform_reparam_transform(fn):
    """
    Return a transform q in (0, 1)^k -> x in parameter space for one sample site.

    Supported automatically:
      - distributions with ``icdf``
      - ``TransformedDistribution``
      - ``Independent`` / ``ExpandedDistribution`` / ``MaskedDistribution``
      - ``MultivariateNormal``

    Unsupported distributions can be passed via ``site_overrides``.
    """
    if isinstance(fn, dist.TransformedDistribution):
        outer = dist.transforms.ComposeTransform(fn.transforms)
        inner = _uniform_reparam_transform(fn.base_dist)
        return lambda q: outer(inner(q))

    if isinstance(fn, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)):
        return _uniform_reparam_transform(fn.base_dist)

    if isinstance(fn, dist.MultivariateNormal):
        affine = dist.transforms.LowerCholeskyAffine(fn.loc, fn.scale_tril)
        return lambda q: affine(dist.Normal(0.0, 1.0).icdf(_safe_unit_interval(q)))

    if isinstance(fn, (dist.BernoulliLogits, dist.BernoulliProbs)):
        probs = jnp.asarray(fn.probs)
        return lambda q: (q < probs).astype(jnp.result_type(probs, int))

    if isinstance(fn, (dist.CategoricalLogits, dist.CategoricalProbs)):
        probs = jnp.asarray(fn.probs)
        return lambda q: jnp.sum(jnp.cumsum(probs, axis=-1) < q[..., None], axis=-1)

    if isinstance(fn, dist.Dirichlet):
        raise NotImplementedError(
            "Automatic transform for Dirichlet is not implemented. "
            "Pass site_overrides={'site_name': custom_transform}."
        )

    if hasattr(fn, "icdf"):
        return lambda q: fn.icdf(_safe_unit_interval(q))

    raise NotImplementedError(
        f"Automatic prior transform is not implemented for {type(fn).__name__}. "
        "Pass site_overrides={'site_name': custom_transform}."
    )


def _get_latent_site_info(model, *, model_args=(), model_kwargs=None):
    model_args = tuple(model_args)
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

    tr = trace(seed(model, random.PRNGKey(0))).get_trace(*model_args, **model_kwargs)

    site_info = []
    for name, site in tr.items():
        if site["type"] != "sample":
            continue
        if site["is_observed"]:
            continue
        if site.get("infer", {}).get("enumerate", "") == "parallel":
            continue

        shape, size = _site_shape_and_size(site["value"])
        site_info.append(
            {
                "name": name,
                "fn": site["fn"],
                "shape": shape,
                "size": size,
            }
        )

    return site_info


class MultiNestRunner:
    """
    Generic PyMultiNest wrapper for a NumPyro model.

    Parameters
    ----------
    model : callable
        NumPyro model.
    model_args : tuple, optional
        Positional arguments forwarded to the model.
    model_kwargs : dict, optional
        Keyword arguments forwarded to the model.
    site_overrides : dict, optional
        Mapping ``{site_name: transform}``, where ``transform(u)`` maps
        unit-cube values to the corresponding site's parameter value.
    """

    def __init__(self, model, *, model_args=(), model_kwargs=None, site_overrides=None):
        self.model = model
        self.model_args = tuple(model_args)
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.site_overrides = {} if site_overrides is None else dict(site_overrides)

        self.logprior, self.loglik = build_logprob_functions(
            model,
            model_args=self.model_args,
            model_kwargs=self.model_kwargs,
        )

        self.site_info = _get_latent_site_info(
            model,
            model_args=self.model_args,
            model_kwargs=self.model_kwargs,
        )
        self.site_names = [s["name"] for s in self.site_info]
        self.param_names = list(self.site_names)
        self.ndim = sum(s["size"] for s in self.site_info)

        self.transforms = {}
        for s in self.site_info:
            name = s["name"]
            if name in self.site_overrides:
                self.transforms[name] = self.site_overrides[name]
            else:
                self.transforms[name] = _uniform_reparam_transform(s["fn"])

        self.outputfiles_basename = None
        self.result = None

    @classmethod
    def from_numpyro_model(cls, model, *, model_args=(), model_kwargs=None, site_overrides=None):
        return cls(
            model,
            model_args=model_args,
            model_kwargs=model_kwargs,
            site_overrides=site_overrides,
        )

    def prior(self, cube):
        cube = jnp.asarray(cube)
        theta_dict = {}
        i = 0
        for s in self.site_info:
            n = s["size"]
            q = cube[i:i + n].reshape(s["shape"])
            theta_dict[s["name"]] = self.transforms[s["name"]](q)
            i += n
        return np.asarray(_flatten_theta_dict(theta_dict, self.site_info), dtype=float)

    def loglikelihood(self, theta):
        theta_dict = _unflatten_theta_vector(theta, self.site_info)
        return float(self.loglik(theta_dict))

    def logposterior(self, theta):
        theta_dict = _unflatten_theta_vector(theta, self.site_info)
        return float(self.loglik(theta_dict) + self.logprior(theta_dict))

    def run(
        self,
        *,
        outputfiles_basename,
        n_live_points=None,
        use_num_live_points_factor=25,
        const_efficiency_mode=True,
        sampling_efficiency=0.03,
        importance_nested_sampling=False,
        multimodal=True,
        resume=True,
        **solve_kwargs,
    ):
        solve, _ = _import_pymultinest()
        self.outputfiles_basename = outputfiles_basename

        if n_live_points is None:
            n_live_points = max(use_num_live_points_factor * max(self.ndim, 1), 100)

        self.result = solve(
            LogLikelihood=self.loglikelihood,
            Prior=self.prior,
            n_dims=self.ndim,
            outputfiles_basename=outputfiles_basename,
            n_live_points=n_live_points,
            const_efficiency_mode=const_efficiency_mode,
            sampling_efficiency=sampling_efficiency,
            importance_nested_sampling=importance_nested_sampling,
            multimodal=multimodal,
            resume=resume,
            **solve_kwargs,
        )
        return self.result

    def get_analyzer(self):
        _, Analyzer = _import_pymultinest()
        if self.outputfiles_basename is None:
            raise RuntimeError("run() must be called before analysis.")
        return Analyzer(n_params=self.ndim, outputfiles_basename=self.outputfiles_basename)

    def get_best_fit(self):
        analyzer = self.get_analyzer()
        theta = np.asarray(analyzer.get_best_fit()["parameters"])
        return _unflatten_theta_vector(theta, self.site_info)

    def get_equal_weighted_posterior(self):
        analyzer = self.get_analyzer()
        arr = np.asarray(analyzer.get_equal_weighted_posterior())[:, :-1]
        return _split_posterior_array(arr, self.site_info)

    def summary(self, *, print_modes=True, make_idata=True):
        analyzer = self.get_analyzer()

        if print_modes:
            modes = analyzer.get_mode_stats()["modes"]
            for mode in modes:
                print(f"\nMode {mode['index']} parameters:")
                mu = _unflatten_theta_vector(np.asarray(mode["mean"]), self.site_info)
                sd = _unflatten_theta_vector(np.asarray(mode["sigma"]), self.site_info)

                for name in self.site_names:
                    x = np.asarray(mu[name])
                    s = np.asarray(sd[name])
                    if x.ndim == 0:
                        print(f"    {name} = {float(x):.6e} +- {float(s):.6e}")
                    else:
                        for idx in np.ndindex(x.shape):
                            idx_str = ",".join(map(str, idx))
                            print(
                                f"    {name}[{idx_str}] = "
                                f"{float(x[idx]):.6e} +- {float(s[idx]):.6e}"
                            )

                print(
                    "    local log-evidence: "
                    f"{mode['local log-evidence']:.3f} +- "
                    f"{mode['local log-evidence error']:.3f}"
                )

        best_fit = self.get_best_fit()
        samples = self.get_equal_weighted_posterior()

        idata = None
        if make_idata and az is not None:
            idata = az.from_dict(posterior={k: v[None, ...] for k, v in samples.items()})

        return best_fit, samples, idata
