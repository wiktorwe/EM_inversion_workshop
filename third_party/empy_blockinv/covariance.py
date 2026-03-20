"""Model covariance and uncertainty helpers."""

from __future__ import annotations

import numpy as np

from .interfaces import InversionResult
from .transforms import LogTransform


def covariance_from_jacobian(
    jacobian_residual_model: np.ndarray,
    model: np.ndarray,
    lam: float,
    ridge_epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate model covariance and correlation from final Jacobian.

    Returns
    -------
    std_log:
        Standard deviations in transformed (log) model space.
    corr_log:
        Correlation matrix in transformed (log) model space.
    cov_log:
        Full covariance matrix in transformed (log) model space.
    """
    model = np.asarray(model, dtype=float)
    jx = np.asarray(jacobian_residual_model, dtype=float)
    t_m = LogTransform()
    dx_dmt = 1.0 / t_m.deriv(model)  # for log transform this is model itself
    jt = jx * dx_dmt[np.newaxis, :]
    hessian = jt.T @ jt + (lam + ridge_epsilon) * np.eye(model.size)
    cov = np.linalg.pinv(hessian)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    safe = np.where(std > 0, std, 1.0)
    corr = cov / np.outer(safe, safe)
    return std, corr, cov


def uncertainty_from_result(
    result: InversionResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical-space absolute and relative parameter uncertainties.

    The inversion is solved in log-parameter space, so we use first-order error
    propagation: sigma_x ~ x * sigma_log.
    """
    if result.final_state is None:
        raise ValueError("Result has no final_state; cannot compute covariance.")
    std_log, _, _ = covariance_from_jacobian(
        jacobian_residual_model=result.final_state.jacobian_residual_model,
        model=result.model,
        lam=result.final_state.lam,
    )
    sigma_abs = result.model * std_log
    sigma_rel = std_log
    return sigma_abs, sigma_rel

