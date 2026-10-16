import functools
from jax import random
from numpyro import handlers, infer
from numpyro.distributions.transforms import biject_to


def _trace_at_constrained_params(model, params_constrained, *model_args, **model_kwargs):
    """Discover sites at supplied latent values, recomputing deterministic sites."""
    def substitute_latent(site):
        if site["type"] == "sample" and not site["is_observed"]:
            return params_constrained.get(site["name"])
        return None

    substituted = handlers.substitute(model, substitute_fn=substitute_latent)
    return handlers.trace(
        handlers.seed(substituted, random.PRNGKey(0))
    ).get_trace(*model_args, **model_kwargs)


def to_unconstrained_dict(model, params_constrained, keys, *model_args, **model_kwargs):
    """
    Convert constrained parameter values into unconstrained representations
    using the model's sample-site supports.

    The model is traced at the supplied constrained latent values to inspect
    sample-site supports, including bounds that depend on other parameters.
    Bijectors are constructed from `site["fn"].support`, and the inverse
    transform is applied to the requested values. Extra deterministic and
    observed entries are ignored; unspecified latent sites are sampled.

    Args:
        model (callable): NumPyro model function.
        params_constrained (dict): Mapping from parameter names to constrained
            values (e.g., positive scales, simplex). Include parameters on which
            the requested sites' supports depend, even if they are not in `keys`.
        keys (iterable[str]): Parameter names to convert into unconstrained space.
        *model_args: Positional arguments passed to `model` when tracing.
        **model_kwargs: Keyword arguments passed to `model` when tracing.

    Returns:
        dict: Mapping from each key in `keys` to its unconstrained value
        (JAX array).

    Raises:
        KeyError: If a key in `keys` does not correspond to a sample site.
    """
    tr = _trace_at_constrained_params(
        model, params_constrained, *model_args, **model_kwargs
    )

    bij = {}
    for name, site in tr.items():
        if site["type"] == "sample" and not site["is_observed"]:
            bij[name] = biject_to(site["fn"].support)

    return {k: bij[k].inv(params_constrained[k]) for k in keys}


def _to_unconstrained(model, params_constrained, keys, *model_args, **model_kwargs):
    """
    Internal alias for backward compatibility.

    Args/Returns:
        Same as `to_unconstrained_dict`.
    """
    return to_unconstrained_dict(model, params_constrained, keys, *model_args, **model_kwargs)


def seed_and_substitute(model, params_dict, param_space, rng_key):
    """
    Seed and substitute parameter values into a NumPyro model.

    Depending on `param_space`, values are interpreted as constrained or
    unconstrained and substituted accordingly.

    Args:
        model (callable): NumPyro model function.
        params_dict (dict): Mapping from parameter names to parameter values.
        param_space (str): Either "unconstrained" or "constrained".
            If "unconstrained", values live in unconstrained space and are
            mapped using NumPyro's `_unconstrain_reparam`. If "constrained",
            values are substituted directly.
        rng_key (jax.random.PRNGKey): RNG key used to seed the model.

    Returns:
        callable: A model wrapped with `handlers.substitute` and `handlers.seed`.

    Raises:
        ValueError: If `param_space` is not "constrained" or "unconstrained".
    """
    if param_space == "unconstrained":
        substituted = handlers.substitute(
            model,
            substitute_fn=functools.partial(
                infer.util._unconstrain_reparam, params_dict
            ),
        )
    elif param_space == "constrained":
        substituted = handlers.substitute(model, data=params_dict)
    else:
        raise ValueError(
            "param_space must be 'constrained' or 'unconstrained'.")

    return handlers.seed(substituted, rng_seed=rng_key)


def _seed_and_substitute(model, params_dict, param_space, rng_key):
    """
    Internal alias for backward compatibility.

    Args/Returns:
        Same as `seed_and_substitute`.
    """
    return seed_and_substitute(model, params_dict, param_space, rng_key)
