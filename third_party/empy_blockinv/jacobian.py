"""Finite-difference Jacobian utilities.

The default perturbation follows pyGIMLi's ModellingBase:
`model[i] *= 1.05` (configurable via `perturb_factor`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .interfaces import ForwardFunction, MisfitFunction
from .residuals import default_residual


def _fd_column(
    idx: int,
    model: np.ndarray,
    base_vec: np.ndarray,
    eval_fn: Any,
    perturb_factor: float,
) -> tuple[int, np.ndarray]:
    perturbed = model.copy()
    perturbed[idx] *= perturb_factor
    delta = perturbed[idx] - model[idx]
    if abs(delta) <= 0.0:
        return idx, np.zeros_like(base_vec)
    changed = np.asarray(eval_fn(perturbed), dtype=float)
    return idx, (changed - base_vec) / delta


def finite_difference_jacobian(
    model: np.ndarray,
    base_vec: np.ndarray,
    eval_fn: Any,
    perturb_factor: float = 1.05,
    parallel: bool = False,
    workers: int | None = None,
) -> np.ndarray:
    """Build Jacobian of an arbitrary vector-valued function via FD columns."""
    n_model = model.size
    n_data = base_vec.size
    jac = np.zeros((n_data, n_model), dtype=float)

    if parallel and n_model > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_fd_column, i, model, base_vec, eval_fn, perturb_factor)
                for i in range(n_model)
            ]
            for fut in futures:
                i, col = fut.result()
                jac[:, i] = col
        return jac

    for i in range(n_model):
        _, col = _fd_column(i, model, base_vec, eval_fn, perturb_factor)
        jac[:, i] = col
    return jac


def response_and_residual(
    model: np.ndarray,
    observed: np.ndarray,
    error: np.ndarray,
    forward_fn: ForwardFunction,
    misfit_fn: MisfitFunction | None,
    callback_context: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute response and residual for the current model."""
    response = np.asarray(forward_fn(model, callback_context), dtype=float)
    if misfit_fn is None:
        residual = default_residual(observed, response, error)
    else:
        residual = np.asarray(
            misfit_fn(observed, response, error, callback_context), dtype=float
        )
    return response, residual

