from jax import jit, random
from numpyro.handlers import seed, substitute, trace


__all__ = ["build_logprob_functions"]


def build_logprob_functions(model, *, model_args=(), model_kwargs=None):
    """
    Construct log-prior and log-likelihood functions for a NumPyro model.

    This wraps a NumPyro model in ``seed`` and ``substitute``, evaluates the
    corresponding trace, and extracts contributions from sample sites to form
    log-prior and log-likelihood terms.

    Args:
        model:
            A NumPyro model function.
        model_args:
            Positional arguments passed to ``model`` when generating the trace.
        model_kwargs:
            Keyword arguments passed to ``model`` when generating the trace.
            Defaults to ``{}``.

    Returns:
        (logprior, loglik):
            A pair of JIT-compiled functions.

            * ``logprior(theta_dict)`` computes the sum of log-probabilities from
              non-observed sample sites.
            * ``loglik(theta_dict)`` computes the sum of log-probabilities from
              observed sample sites.

    Notes:
        The model is run with a fixed PRNG key and with values provided by
        ``theta_dict`` substituted at the corresponding sample sites.

        Contributions added via ``numpyro.factor`` are treated as observed-site
        terms and are therefore included in ``loglik``, not in ``logprior``.
    """
    model_args = tuple(model_args)
    if model_kwargs is None:
        model_kwargs = {}

    def run_model(theta_dict):
        seeded = seed(model, random.PRNGKey(0))
        substituted = substitute(seeded, data=theta_dict)
        return trace(substituted).get_trace(*model_args, **model_kwargs)

    def _sum_logprob(tr, observed):
        total = 0.0
        for _, site in tr.items():
            if site["type"] == "sample" and site["is_observed"] == observed:
                total = total + site["fn"].log_prob(site["value"]).sum()
        return total

    def logprior(theta_dict):
        tr = run_model(theta_dict)
        return _sum_logprob(tr, observed=False)

    def loglik(theta_dict):
        tr = run_model(theta_dict)
        return _sum_logprob(tr, observed=True)

    return jit(logprior), jit(loglik)
