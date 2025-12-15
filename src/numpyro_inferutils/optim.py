import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.infer.initialization import init_to_value, init_to_sample


def find_map_svi(
    numpyro_model,
    step_size,
    num_steps,
    *,
    rng_key,
    p_initial=None,
    progress_bar=True,
    **kwargs,
):
    """Optimization via SVI with a Laplace autoguide (MAP estimate).

    This function uses AutoLaplaceApproximation and returns only the
    MAP-like point estimate (guide median). The covariance of the
    Laplace approximation is intentionally not used, as it may be
    unstable for some models. This is primarily intended to provide
    an initial point for NUTS.

    Args:
        numpyro_model: NumPyro model.
        step_size: Step size for Adam.
        num_steps: Number of SVI steps.
        rng_key: PRNG key passed to svi.run.
        p_initial: Optional dict of initial values at sample sites
            (constrained space). If None, init_to_sample is used.
        progress_bar: Whether to show the SVI progress bar.
        **kwargs: Passed to the SVI constructor.

    Returns:
        dict: MAP parameter values (constrained space).
    """
    optimizer = numpyro.optim.Adam(step_size=step_size)

    if p_initial is None:
        guide = AutoLaplaceApproximation(
            numpyro_model, init_loc_fn=init_to_sample
        )
    else:
        guide = AutoLaplaceApproximation(
            numpyro_model, init_loc_fn=init_to_value(values=p_initial)
        )

    svi = SVI(
        numpyro_model,
        guide,
        optimizer,
        loss=Trace_ELBO(),
        **kwargs,
    )

    svi_result = svi.run(
        rng_key,
        num_steps,
        progress_bar=progress_bar,
    )

    params_svi = svi_result.params
    p_fit = guide.median(params_svi)

    return p_fit
