"""Linearized solvers for Marquardt/Gauss-Newton style model updates."""

from __future__ import annotations

import numpy as np


def solve_linear_system(a_mat: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    """Solve linear system with SciPy fallback to NumPy.

    We prefer SciPy for robustness/conditioning controls, but keep a NumPy
    fallback so the package remains usable in lighter environments.
    """
    try:
        import scipy.linalg  # type: ignore

        return scipy.linalg.solve(a_mat, b_vec, assume_a="sym")
    except Exception:
        try:
            return np.linalg.solve(a_mat, b_vec)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
            return sol


def marquardt_step(
    residual: np.ndarray,
    jacobian_transformed: np.ndarray,
    model_transformed: np.ndarray,
    lam: float,
    reference_transformed: np.ndarray,
    ridge_epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute one transformed-space Marquardt step.

    This mirrors cType=0 damping in transformed space:
    minimize ||r||^2 + lam * ||(m_t - m_ref_t)||^2 (linearized update).
    """
    jt = jacobian_transformed.T
    jtj = jt @ jacobian_transformed
    reg = np.eye(model_transformed.size, dtype=float)
    a_mat = jtj + (lam + ridge_epsilon) * reg
    grad = jt @ residual + lam * (model_transformed - reference_transformed)
    return solve_linear_system(a_mat, -grad)

