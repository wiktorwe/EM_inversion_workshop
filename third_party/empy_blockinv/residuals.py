"""Residual and misfit helpers."""

from __future__ import annotations

import numpy as np


def default_residual(
    observed: np.ndarray,
    predicted: np.ndarray,
    error: np.ndarray,
) -> np.ndarray:
    """Return weighted residual `(obs - pred) / err`.

    The caller is responsible for providing `error` in the same unit as
    `observed` and `predicted` (for phase this is typically absolute error).
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    err = np.asarray(error, dtype=float)
    if obs.shape != pred.shape or obs.shape != err.shape:
        raise ValueError("observed, predicted and error must share shape.")
    if np.any(err <= 0):
        raise ValueError("error entries must be > 0.")
    return (obs - pred) / err


def phi_data(residual: np.ndarray) -> float:
    """Return data objective function value `phiD = ||r||^2`."""
    r = np.asarray(residual, dtype=float)
    return float(r @ r)


def chi2(residual: np.ndarray) -> float:
    """Return chi-square style data misfit `phiD / n_data`."""
    r = np.asarray(residual, dtype=float)
    return float((r @ r) / r.size)

