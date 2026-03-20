"""High-level block1D inversion runner for pluggable Empymod workflows."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .interfaces import (
    ForwardFunction,
    InversionConfig,
    InversionResult,
    InversionState,
    MisfitFunction,
)
from .jacobian import finite_difference_jacobian, response_and_residual
from .linesearch import line_search_tau
from .model import create_start_model, parameter_count, validate_positive_model
from .regularization import phi_model_c0
from .residuals import chi2, phi_data
from .solver import marquardt_step
from .transforms import LogTransform


def _ensure_start_model(
    config: InversionConfig,
    observed: np.ndarray,
    start_depth_min: float | None,
    start_depth_max: float | None,
) -> np.ndarray:
    if config.start_model is not None:
        model = np.asarray(config.start_model, dtype=float)
    else:
        if start_depth_min is None or start_depth_max is None:
            raise ValueError(
                "Provide config.start_model or both start_depth_min/start_depth_max."
            )
        model = create_start_model(
            data=observed,
            n_layers=config.n_layers,
            min_depth=float(start_depth_min),
            max_depth=float(start_depth_max),
        )
    expected = parameter_count(config.n_layers)
    if model.size != expected:
        raise ValueError(f"start_model size {model.size} != expected {expected}.")
    validate_positive_model(model)
    return model


def run_block1d_inversion(
    observed: np.ndarray,
    error: np.ndarray,
    config: InversionConfig,
    forward_fn: ForwardFunction,
    misfit_fn: MisfitFunction | None = None,
    *,
    start_depth_min: float | None = None,
    start_depth_max: float | None = None,
) -> InversionResult:
    """Run block1D inversion with pyGIMLi-inspired control flow.

    Notes
    -----
    - Model updates happen in transformed (log) space.
    - Damping regularization is cType=0-like in transformed space.
    - Jacobians are finite-difference, with multiplicative perturbations.
    - `misfit_fn` is optional. If omitted, residual is `(obs-pred)/error`.
    """
    obs = np.asarray(observed, dtype=float)
    err = np.asarray(error, dtype=float)
    if obs.shape != err.shape:
        raise ValueError("observed and error must have identical shape.")
    if np.any(err <= 0):
        raise ValueError("error values must be > 0.")

    model = _ensure_start_model(
        config=config,
        observed=obs,
        start_depth_min=start_depth_min,
        start_depth_max=start_depth_max,
    )
    transform = LogTransform(config.lower_bound or 0.0, config.upper_bound)
    model_t = transform.trans(model)

    if config.use_reference_model:
        if config.reference_model is None:
            raise ValueError("reference_model required when use_reference_model=True.")
        ref_model = np.asarray(config.reference_model, dtype=float)
        if ref_model.size != model.size:
            raise ValueError("reference_model size mismatch.")
        validate_positive_model(ref_model)
        reference_t = transform.trans(ref_model)
    else:
        reference_t = np.zeros_like(model_t)

    # Iteration histories mirror what pyGIMLi exposes for diagnostics.
    chi2_history: list[float] = []
    phi_history: list[float] = []
    lambda_history: list[float] = []
    model_history: list[np.ndarray] = [model.copy()]

    response, residual = response_and_residual(
        model=model,
        observed=obs,
        error=err,
        forward_fn=forward_fn,
        misfit_fn=misfit_fn,
        callback_context=config.callback_context,
    )
    phi_d = phi_data(residual)
    phi_m = phi_model_c0(
        model_transformed=model_t,
        use_reference_model=config.use_reference_model,
        reference_transformed=reference_t,
    )
    if config.local_regularization:
        phi_total = phi_d
    else:
        phi_total = phi_d + config.lam * phi_m

    chi2_history.append(chi2(residual))
    phi_history.append(phi_total)
    lambda_history.append(config.lam)

    last_state: InversionState | None = None
    lam = float(config.lam)
    last_phi = float(phi_total)

    for iteration in range(1, config.max_iter + 1):
        # Build response Jacobian wrt physical model first.
        # This follows pyGIMLi's default brute-force Jacobian strategy.
        response_eval = lambda m: np.asarray(
            forward_fn(np.asarray(m, dtype=float), config.callback_context),
            dtype=float,
        )
        jac_data_model = finite_difference_jacobian(
            model=model,
            base_vec=response,
            eval_fn=response_eval,
            perturb_factor=config.perturb_factor,
            parallel=config.fd_parallel,
            workers=config.fd_workers,
        )

        # Build residual Jacobian wrt physical model.
        # For custom misfits this captures full local sensitivity of r(m).
        residual_eval = lambda m: response_and_residual(
            model=np.asarray(m, dtype=float),
            observed=obs,
            error=err,
            forward_fn=forward_fn,
            misfit_fn=misfit_fn,
            callback_context=config.callback_context,
        )[1]
        jac_res_model = finite_difference_jacobian(
            model=model,
            base_vec=residual,
            eval_fn=residual_eval,
            perturb_factor=config.perturb_factor,
            parallel=config.fd_parallel,
            workers=config.fd_workers,
        )

        # Convert model derivatives to transformed-space derivatives:
        # J_t = J_x * dx/dm_t and for log transform dx/dm_t = x.
        dx_dmt = 1.0 / transform.deriv(model)
        jac_res_trans = jac_res_model * dx_dmt[np.newaxis, :]

        dmodel_t = marquardt_step(
            residual=residual,
            jacobian_transformed=jac_res_trans,
            model_transformed=model_t,
            lam=lam,
            reference_transformed=reference_t,
            ridge_epsilon=config.ridge_epsilon,
        )

        # Candidate full-step model.
        model_new = transform.update(model, dmodel_t)
        model_new = np.asarray(model_new, dtype=float)
        # Hard clip if bounds are provided.
        if config.lower_bound is not None:
            model_new = np.maximum(model_new, config.lower_bound)
        if config.upper_bound is not None:
            model_new = np.minimum(model_new, config.upper_bound)
        validate_positive_model(model_new)

        response_new, residual_new = response_and_residual(
            model=model_new,
            observed=obs,
            error=err,
            forward_fn=forward_fn,
            misfit_fn=misfit_fn,
            callback_context=config.callback_context,
        )

        if config.line_search:
            dmt = transform.trans(model_new) - model_t

            def objective_for_tau(tau: float) -> float:
                trial_t = model_t + tau * dmt
                trial = transform.inv_trans(trial_t)
                if config.lower_bound is not None:
                    trial = np.maximum(trial, config.lower_bound)
                if config.upper_bound is not None:
                    trial = np.minimum(trial, config.upper_bound)
                pred, res = response_and_residual(
                    model=trial,
                    observed=obs,
                    error=err,
                    forward_fn=forward_fn,
                    misfit_fn=misfit_fn,
                    callback_context=config.callback_context,
                )
                # pred is intentionally unused here; objective comes from residual.
                _ = pred
                pd = phi_data(res)
                if config.local_regularization:
                    return pd
                pm = phi_model_c0(
                    model_transformed=trial_t,
                    use_reference_model=config.use_reference_model,
                    reference_transformed=reference_t,
                )
                return pd + lam * pm

            tau = line_search_tau(
                objective_for_tau=objective_for_tau,
                local_regularization=config.local_regularization,
            )
            if tau < 0.95:
                model_new = transform.inv_trans(model_t + tau * dmt)
                if config.lower_bound is not None:
                    model_new = np.maximum(model_new, config.lower_bound)
                if config.upper_bound is not None:
                    model_new = np.minimum(model_new, config.upper_bound)
                validate_positive_model(model_new)
                response_new, residual_new = response_and_residual(
                    model=model_new,
                    observed=obs,
                    error=err,
                    forward_fn=forward_fn,
                    misfit_fn=misfit_fn,
                    callback_context=config.callback_context,
                )

        # Accept update.
        model = model_new
        model_t = transform.trans(model)
        response = response_new
        residual = residual_new

        phi_d = phi_data(residual)
        phi_m = phi_model_c0(
            model_transformed=model_t,
            use_reference_model=config.use_reference_model,
            reference_transformed=reference_t,
        )
        if config.local_regularization:
            phi_total = phi_d
        else:
            phi_total = phi_d + lam * phi_m

        chi2_now = chi2(residual)
        chi2_history.append(chi2_now)
        phi_history.append(phi_total)
        lambda_history.append(lam)
        model_history.append(model.copy())

        # Store complete final-state information for covariance and diagnostics.
        last_state = InversionState(
            model=model.copy(),
            response=response.copy(),
            residual=residual.copy(),
            chi2=chi2_now,
            phi_data=phi_d,
            phi_model=phi_m,
            lam=lam,
            iteration=iteration,
            jacobian_data_model=jac_data_model,
            jacobian_residual_model=jac_res_model,
        )

        # Stop criteria mirror pyGIMLi flow.
        if config.stop_at_chi1 and chi2_now <= 1.0:
            break
        if iteration > 2 and (phi_total / last_phi) > (1.0 - config.min_dphi_percent / 100.0):
            break
        last_phi = phi_total
        lam = max(config.min_lambda, lam * config.lambda_factor)

    result = InversionResult(
        model=model.copy(),
        response=response.copy(),
        residual=residual.copy(),
        chi2_history=chi2_history,
        phi_history=phi_history,
        lambda_history=lambda_history,
        model_history=model_history,
        final_state=last_state,
    )
    return result


def with_start_model(config: InversionConfig, model: np.ndarray) -> InversionConfig:
    """Convenience helper returning a config copy with a new start model."""
    return replace(config, start_model=np.asarray(model, dtype=float))

