__all__ = ["MultiNestRunner"]

import numpy as np
import jax.numpy as jnp
from jax import random
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive
import numpyro.distributions as dist
import warnings

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
      - ``Bernoulli`` / ``Categorical``
      - ``Gamma`` (with dtype stabilization)

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

    if isinstance(fn, dist.Gamma):
        # NumPyro's Gamma.icdf can fail when concentration/rate are integer typed,
        # because the underlying TFP helper expects a floating common dtype.
        dtype = jnp.result_type(fn.concentration, fn.rate, float)
        gamma = dist.Gamma(
            jnp.asarray(fn.concentration, dtype=dtype),
            jnp.asarray(fn.rate, dtype=dtype),
        )
        return lambda q: gamma.icdf(_safe_unit_interval(q))

    if hasattr(fn, "icdf"):
        return lambda q: fn.icdf(_safe_unit_interval(q))

    raise NotImplementedError(
        f"Automatic prior transform is not implemented for {type(fn).__name__}. "
        "Pass site_overrides={'site_name': custom_transform}."
    )


def _get_trace(model, *, model_args=(), model_kwargs=None):
    model_args = tuple(model_args)
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    return trace(seed(model, random.PRNGKey(0))).get_trace(*model_args, **model_kwargs)


def _get_latent_site_info(model, *, model_args=(), model_kwargs=None):
    tr = _get_trace(model, model_args=model_args, model_kwargs=model_kwargs)

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


def _get_deterministic_site_names(model, *, model_args=(), model_kwargs=None):
    tr = _get_trace(model, model_args=model_args, model_kwargs=model_kwargs)
    return [name for name, site in tr.items() if site["type"] == "deterministic"]


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
        self.site_overrides = {} if site_overrides is None else dict(
            site_overrides)

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
        self.deterministic_site_names = _get_deterministic_site_names(
            model,
            model_args=self.model_args,
            model_kwargs=self.model_kwargs,
        )

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
            name = s["name"]
            try:
                theta_dict[name] = self.transforms[name](q)
            except Exception as e:
                raise RuntimeError(
                    f"Error in prior transform for site {name!r} "
                    f"({type(s['fn']).__name__}, shape={s['shape']}): {e}"
                ) from e
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
        const_efficiency_mode=False,
        sampling_efficiency=None,
        importance_nested_sampling=False,
        multimodal=True,
        resume=True,
        **solve_kwargs,
    ):
        solve, _ = _import_pymultinest()
        self.outputfiles_basename = outputfiles_basename

        if n_live_points is None:
            n_live_points = max(
                use_num_live_points_factor * max(self.ndim, 1), 100)

        if sampling_efficiency is None:
            sampling_efficiency = 0.03 if const_efficiency_mode else 0.8

        # Dry-run once outside the ctypes callback for clearer Python errors.
        _ = self.prior(np.full(self.ndim, 0.5, dtype=float))

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

    def _normalize_requested_deterministic_sites(self, site_names):
        if site_names is None:
            return []
        if isinstance(site_names, str):
            site_names = [site_names]

        requested = list(site_names)
        deterministic_set = set(self.deterministic_site_names)
        sample_set = set(self.site_names)

        selected = [name for name in requested if name in deterministic_set]
        unknown = sorted(
            name
            for name in requested
            if name not in deterministic_set and name not in sample_set
        )

        if unknown:
            warnings.warn(
                "Ignoring unknown site name(s) in deterministic_sites: "
                f"{unknown}. Available deterministic sites: "
                f"{self.deterministic_site_names}",
                stacklevel=2,
            )

        return selected

    def get_deterministic_samples(self, posterior_samples, *, site_names):
        site_names = self._normalize_requested_deterministic_sites(site_names)
        if not site_names:
            return {}

        predictive = Predictive(
            self.model,
            posterior_samples=posterior_samples,
            return_sites=site_names,
        )
        out = predictive(random.PRNGKey(
            0), *self.model_args, **self.model_kwargs)
        return {name: out[name] for name in site_names}

    def get_deterministic_best_fit(self, *, site_names):
        site_names = self._normalize_requested_deterministic_sites(site_names)
        if not site_names:
            return {}

        best_fit = self.get_best_fit()
        predictive = Predictive(
            self.model,
            posterior_samples={k: jnp.expand_dims(
                v, axis=0) for k, v in best_fit.items()},
            return_sites=site_names,
        )
        out = predictive(random.PRNGKey(
            0), *self.model_args, **self.model_kwargs)
        return {name: out[name][0] for name in site_names}

    def summary(self, *, print_modes=True, make_idata=True, deterministic_sites=None):
        analyzer = self.get_analyzer()

        if print_modes:
            modes = analyzer.get_mode_stats()["modes"]
            for mode in modes:
                print(f"\nMode {mode['index']} parameters:")
                mu = _unflatten_theta_vector(
                    np.asarray(mode["mean"]), self.site_info)
                sd = _unflatten_theta_vector(
                    np.asarray(mode["sigma"]), self.site_info)

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

        if deterministic_sites is not None:
            det_best_fit = self.get_deterministic_best_fit(
                site_names=deterministic_sites)
            det_samples = self.get_deterministic_samples(
                samples, site_names=deterministic_sites)
            best_fit = {**best_fit, **det_best_fit}
            samples = {**samples, **det_samples}

            if print_modes and det_samples:
                print("\nDeterministic sites (posterior samples):")
                for name, vals in det_samples.items():
                    x = np.asarray(vals)
                    mu = np.mean(x, axis=0)
                    sd = np.std(
                        x, axis=0, ddof=1) if x.shape[0] > 1 else np.zeros_like(mu)
                    if mu.ndim == 0:
                        print(
                            f"    {name} = {float(mu):.6e} +- {float(sd):.6e}")
                    else:
                        for idx in np.ndindex(mu.shape):
                            idx_str = ",".join(map(str, idx))
                            print(
                                f"    {name}[{idx_str}] = "
                                f"{float(mu[idx]):.6e} +- {float(sd[idx]):.6e}"
                            )

        idata = None
        if make_idata and az is not None:
            idata = az.from_dict(posterior={k: np.asarray(
                v)[None, ...] for k, v in samples.items()})

        return best_fit, samples, idata
