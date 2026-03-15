from collections import OrderedDict

import jax.numpy as jnp
from jax import jacfwd, jacrev, random
from numpyro import handlers

from .transforms import _seed_and_substitute, _to_unconstrained


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _concat_blocks(blocks):
    return blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks)


def _prepare_vector(
    x,
    *,
    block_lengths=None,
    name="value",
    allow_scalar_broadcast=False,
):
    """Normalize an input into a 1D vector."""
    if isinstance(x, (list, tuple)) and block_lengths is not None and len(x) == len(block_lengths):
        parts = []
        for i, (xi, n) in enumerate(zip(x, block_lengths)):
            ai = jnp.asarray(xi).reshape(-1)
            if allow_scalar_broadcast and ai.size == 1:
                ai = jnp.broadcast_to(ai, (n,))
            if ai.shape != (n,):
                raise ValueError(
                    f"{name}[{i}] has shape {ai.shape}, expected {(n,)}."
                )
            parts.append(ai)
        return _concat_blocks(parts)

    a = jnp.asarray(x).reshape(-1)
    if block_lengths is not None and allow_scalar_broadcast and a.size == 1:
        a = jnp.broadcast_to(a, (sum(block_lengths),))
    return a


def _trace_model(model, params_dict, param_space, rng_key, *, model_args=(), model_kwargs=None):
    """Run a NumPyro model with substituted parameters and return the trace."""
    model_kwargs = {} if model_kwargs is None else model_kwargs
    seeded = _seed_and_substitute(model, params_dict, param_space, rng_key)
    return handlers.trace(seeded).get_trace(*model_args, **model_kwargs)


def _latent_sample_names(model, *, model_args=(), model_kwargs=None):
    """Return non-observed sample-site names in the model trace."""
    model_kwargs = {} if model_kwargs is None else model_kwargs
    tr = handlers.trace(handlers.seed(model, random.PRNGKey(0))).get_trace(
        *model_args, **model_kwargs
    )
    return [
        name
        for name, site in tr.items()
        if site["type"] == "sample" and not site["is_observed"]
    ]


def _prepare_param_dicts(model, pdic, keys, *, param_space, model_args=(), model_kwargs=None):
    """
    Prepare full and differentiable parameter dicts in the requested parameter space.

    `pdic` is assumed to contain constrained values. When `param_space='unconstrained'`,
    any latent sample sites present in `pdic` are mapped to unconstrained space.
    Non-latent entries are kept as-is.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs
    keys = list(keys)

    if param_space == "unconstrained":
        latent_names = set(
            _latent_sample_names(
                model, model_args=model_args, model_kwargs=model_kwargs)
        )
        convert_keys = [k for k in pdic.keys() if k in latent_names]
        converted = _to_unconstrained(
            model, pdic, convert_keys, *model_args, **model_kwargs
        ) if len(convert_keys) > 0 else {}
        pdic_all = dict(pdic)
        pdic_all.update(converted)
    elif param_space == "constrained":
        pdic_all = dict(pdic)
    else:
        raise ValueError(
            "param_space must be 'constrained' or 'unconstrained'.")

    try:
        pdic_sub = OrderedDict((k, pdic_all[k]) for k in keys)
    except KeyError as e:
        raise KeyError(
            f"Parameter '{e.args[0]}' in `keys` is missing from `pdic`.") from e

    return pdic_all, pdic_sub


def _flatten_jacobian_tree(Jtree, keys):
    """Flatten a Jacobian pytree into a dense matrix with stable column metadata."""
    N = Jtree[keys[0]].shape[0]
    cols, names, slices, c0 = [], [], {}, 0
    for k in keys:
        Jk = jnp.asarray(Jtree[k]).reshape(N, -1)
        cols.append(Jk)
        d = Jk.shape[1]
        names += [k] if d == 1 else [f"{k}[{i}]" for i in range(d)]
        slices[k] = slice(c0, c0 + d)
        c0 += d
    J = jnp.hstack(cols)
    return J, slices, names


def _flatten_hessian_tree(Htree, pdic_sub, keys):
    """Flatten a Hessian pytree-of-pytrees into a dense matrix."""
    sizes = OrderedDict((k, int(jnp.asarray(pdic_sub[k]).size)) for k in keys)

    names, slices, c0 = [], {}, 0
    for k in keys:
        d = sizes[k]
        names += [k] if d == 1 else [f"{k}[{i}]" for i in range(d)]
        slices[k] = slice(c0, c0 + d)
        c0 += d

    rows = []
    for ki in keys:
        di = sizes[ki]
        row_blocks = []
        for kj in keys:
            dj = sizes[kj]
            Hij = jnp.asarray(Htree[ki][kj]).reshape(di, dj)
            row_blocks.append(Hij)
        rows.append(jnp.hstack(row_blocks))

    H = jnp.vstack(rows)
    return H, slices, names


def _sum_logprob(trace, *, observed):
    """Sum log-probabilities over observed or unobserved sample sites."""
    total = 0.0
    for _, site in trace.items():
        if site["type"] == "sample" and site["is_observed"] == observed:
            total = total + site["fn"].log_prob(site["value"]).sum()
    return total


def _objective_from_model(
    model,
    params_dict,
    param_space,
    rng_key,
    *,
    which,
    model_args=(),
    model_kwargs=None,
):
    """
    Evaluate logprior / loglik / logprob from a NumPyro model.

    Contributions from `numpyro.factor` are treated as observed-site terms and
    therefore contribute to `loglik` and `logprob`, not to `logprior`.
    """
    tr = _trace_model(
        model,
        params_dict,
        param_space,
        rng_key,
        model_args=model_args,
        model_kwargs=model_kwargs,
    )

    if which == "logprior":
        return _sum_logprob(tr, observed=False)
    if which == "loglik":
        return _sum_logprob(tr, observed=True)
    if which == "logprob":
        return _sum_logprob(tr, observed=False) + _sum_logprob(tr, observed=True)

    raise ValueError("which must be 'loglik', 'logprior', or 'logprob'.")


def _std_residuals_from_model_independent_normal(
    model,
    params_dict,
    param_space,
    rng_key,
    *,
    sigma_sd,
    mu_name="model",
    obs_name="obs",
    observed=None,
    model_args=(),
    model_kwargs=None,
):
    """
    Build standardized residuals r = (y - mu(theta)) / sigma for an independent
    Gaussian likelihood, using a NumPyro model.

    Args:
        model (callable): NumPyro model.
        params_dict (dict): Mapping from parameter names to values in either
            constrained or unconstrained space.
        param_space (str): Either "constrained" or "unconstrained".
        rng_key (jax.random.PRNGKey): RNG key used to seed the model.
        sigma_sd (array-like or list/tuple of array-like): Standard deviations for
            each data point. If `mu_name` is multiple, this may be one concatenated
            array or one entry per block.
        mu_name (str or list/tuple of str, optional): Deterministic site name(s)
            holding the model mean. If multiple names are given, they are flattened
            and concatenated.
        obs_name (str or list/tuple of str, optional): Observed site name(s) for
            the data. If multiple names are given, they are flattened and concatenated.
        observed (array-like or list/tuple of array-like, optional): Explicit
            observed values; overrides trace values if provided.
        model_args (tuple): Positional arguments for the model.
        model_kwargs (dict or None): Keyword arguments for the model.

    Returns:
        jnp.ndarray: Standardized residuals with shape (N,).

    Raises:
        KeyError: If `mu_name` or `obs_name` is not found in the trace (when required).
        ValueError: If shapes of `y`, `mu`, and `sigma_sd` do not match.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs
    trace = _trace_model(
        model,
        params_dict,
        param_space,
        rng_key,
        model_args=model_args,
        model_kwargs=model_kwargs,
    )

    mu_names = _as_list(mu_name)
    mu_blocks = []
    for name in mu_names:
        if name not in trace:
            raise KeyError(
                f"deterministic mu '{name}' not found in trace. "
                "Record it via numpyro.deterministic(mu_name, mu)."
            )
        mu_blocks.append(jnp.asarray(trace[name]["value"]).reshape(-1))

    block_lengths = [b.shape[0] for b in mu_blocks]
    mu = _concat_blocks(mu_blocks)

    sigma_sd = _prepare_vector(
        sigma_sd,
        block_lengths=block_lengths,
        name="sigma_sd",
        allow_scalar_broadcast=True,
    )

    if observed is not None:
        y = _prepare_vector(
            observed,
            block_lengths=block_lengths,
            name="observed",
            allow_scalar_broadcast=False,
        )
    else:
        if obs_name is None:
            raise ValueError(
                "Either `observed` or `obs_name` must be provided.")
        obs_names = _as_list(obs_name)
        if len(obs_names) == 1:
            name = obs_names[0]
            if name not in trace:
                raise KeyError(f"obs site '{name}' not found in trace.")
            y = jnp.asarray(trace[name]["value"]).reshape(-1)
        else:
            if len(obs_names) != len(block_lengths):
                raise ValueError(
                    "`obs_name` must be a string, or a list/tuple with one "
                    "entry per `mu_name` block."
                )
            obs_blocks = []
            for i, (name, n) in enumerate(zip(obs_names, block_lengths)):
                if name not in trace:
                    raise KeyError(f"obs site '{name}' not found in trace.")
                yb = jnp.asarray(trace[name]["value"]).reshape(-1)
                if yb.shape != (n,):
                    raise ValueError(
                        f"obs block {i} has shape {yb.shape}, expected {(n,)}."
                    )
                obs_blocks.append(yb)
            y = _concat_blocks(obs_blocks)

    if y.shape != mu.shape or y.shape != sigma_sd.shape:
        raise ValueError(
            f"shape mismatch: y {y.shape}, mu {mu.shape}, sigma {sigma_sd.shape}"
        )
    return (y - mu) / sigma_sd  # (N,)


def information_from_model_independent_normal(
    *,
    model=None,
    model_args=(),
    model_kwargs=None,
    pdic=None,
    mu_name=None,
    observed=None,
    obs_name=None,
    keys=None,
    sigma_sd=None,
    param_space="unconstrained",
    rng_key=None,
    diff_mode="fwd",  # usually preferable when N_data >> N_params
):
    """
    Compute Fisher information matrix for an independent Gaussian likelihood
    directly from a NumPyro model, using (observed - mu(pdic)) / sigma_sd.

    Args:
        model: NumPyro model.
        model_args, model_kwargs: static args/kwargs for the model.
        pdic: dict of parameter values in constrained space.
        mu_name: deterministic site name(s) for the model mean. Multiple names
            are flattened and concatenated.
        observed: 1D array or blockwise list/tuple of observed values; `obs_name`
            is used if not provided.
        obs_name: observed site name(s).
        keys: list of parameter names to differentiate (order preserved).
        sigma_sd: 1D array or blockwise list/tuple of standard deviations.
        param_space: 'constrained' or 'unconstrained'; use 'unconstrained' to
            initialize inverse_mass_matrix.
        rng_key: PRNG key (default = jax.random.PRNGKey(0)).
        diff_mode: {'fwd', 'rev'}
            Differentiation mode for computing the Jacobian of the
            standardized residuals with respect to the parameters.
            The default is `'fwd'`, which is often preferable when the
            number of data points is much larger than the number of
            differentiated parameters. Use `'rev'` if forward-mode
            autodiff is unsupported or slower for the model of interest.

    Returns:
        dict:
            - "fisher" (jnp.ndarray): The (P, P) Fisher information matrix.
            - "col_slices" (dict[str, slice]): Mapping from each parameter name
              to its corresponding column range in the Fisher matrix.
            - "col_names" (list[str]): Flattened per-column names.
            - "params_unconstrained" (dict[str, jnp.ndarray]): Parameter values
              in the requested differentiation space used internally.
    """
    assert (
        model is not None
        and pdic is not None
        and mu_name is not None
        and keys is not None
        and sigma_sd is not None
    )
    if (observed is None) and (obs_name is None):
        raise ValueError("Either `observed` or `obs_name` must be provided.")

    keys = list(keys)
    rng_key = random.PRNGKey(0) if rng_key is None else rng_key
    model_kwargs = {} if model_kwargs is None else model_kwargs

    pdic_all, pdic_sub = _prepare_param_dicts(
        model,
        pdic,
        keys,
        param_space=param_space,
        model_args=model_args,
        model_kwargs=model_kwargs,
    )
    base = dict((k, v) for k, v in pdic_all.items() if k not in keys)

    def r_fn(p_sub):
        p_all = dict(base)
        p_all.update(p_sub)
        return _std_residuals_from_model_independent_normal(
            model,
            p_all,
            param_space,
            rng_key,
            sigma_sd=sigma_sd,
            mu_name=mu_name,
            obs_name=obs_name,
            model_args=model_args,
            model_kwargs=model_kwargs,
            observed=observed,
        )

    if diff_mode == "rev":
        jac = jacrev
    elif diff_mode == "fwd":
        jac = jacfwd
    else:
        raise ValueError("diff_mode must be 'rev' or 'fwd'.")

    Jtree = jac(r_fn)(pdic_sub)
    J, slices, names = _flatten_jacobian_tree(Jtree, keys)
    F = J.T @ J
    return {
        "fisher": F,
        "col_slices": slices,
        "col_names": names,
        "params_unconstrained": pdic_all,
    }


def hessian_from_model(
    *,
    model=None,
    model_args=(),
    model_kwargs=None,
    pdic=None,
    keys=None,
    which="logprob",  # "loglik", "logprior", "logprob"
    param_space="unconstrained",
    rng_key=None,
    diff_mode="fwdrev",  # "fwdrev", "revfwd", "revrev", "fwdfwd"
    symmetrize=True,
):
    """
    Compute the Hessian of loglik / logprior / logprob directly from a NumPyro model.

    Args:
        model: NumPyro model.
        model_args, model_kwargs: static args/kwargs for the model.
        pdic: dict of parameter values in constrained space.
        keys: list of parameter names to differentiate (order preserved).
        which: {'loglik', 'logprior', 'logprob'}
            Target scalar objective whose Hessian is computed.
        param_space: 'constrained' or 'unconstrained'.
            Use 'unconstrained' if you want curvature in the unconstrained space.
        rng_key: PRNG key (default = jax.random.PRNGKey(0)).
        diff_mode: {'fwdrev', 'revfwd', 'revrev', 'fwdfwd'}
            Nesting of AD modes for the Hessian. 'fwdrev' is usually a good default.
        symmetrize: if True, return 0.5 * (H + H.T).

    Returns:
        dict:
            - "hessian" (jnp.ndarray): The (P, P) Hessian matrix.
            - "value" (float): Objective value at the supplied parameter point.
            - "col_slices" (dict[str, slice]): Mapping from each parameter name
              to its corresponding column range.
            - "col_names" (list[str]): Flattened per-column names.
            - "params_unconstrained" (dict[str, jnp.ndarray]): Parameter values
              in the requested differentiation space used internally.
    """
    assert model is not None and pdic is not None and keys is not None

    keys = list(keys)
    rng_key = random.PRNGKey(0) if rng_key is None else rng_key
    model_kwargs = {} if model_kwargs is None else model_kwargs

    pdic_all, pdic_sub = _prepare_param_dicts(
        model,
        pdic,
        keys,
        param_space=param_space,
        model_args=model_args,
        model_kwargs=model_kwargs,
    )
    base = dict((k, v) for k, v in pdic_all.items() if k not in keys)

    def objective_fn(p_sub):
        p_all = dict(base)
        p_all.update(p_sub)
        return _objective_from_model(
            model,
            p_all,
            param_space,
            rng_key,
            which=which,
            model_args=model_args,
            model_kwargs=model_kwargs,
        )

    if diff_mode == "fwdrev":
        Htree = jacfwd(jacrev(objective_fn))(pdic_sub)
    elif diff_mode == "revfwd":
        Htree = jacrev(jacfwd(objective_fn))(pdic_sub)
    elif diff_mode == "revrev":
        Htree = jacrev(jacrev(objective_fn))(pdic_sub)
    elif diff_mode == "fwdfwd":
        Htree = jacfwd(jacfwd(objective_fn))(pdic_sub)
    else:
        raise ValueError(
            "diff_mode must be 'fwdrev', 'revfwd', 'revrev', or 'fwdfwd'."
        )

    H, slices, names = _flatten_hessian_tree(Htree, pdic_sub, keys)
    if symmetrize:
        H = 0.5 * (H + H.T)

    return {
        "hessian": H,
        "value": objective_fn(pdic_sub),
        "col_slices": slices,
        "col_names": names,
        "params_unconstrained": pdic_all,
    }
