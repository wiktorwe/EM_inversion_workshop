"""Utilities for 1D layered block-model parameterization."""

from __future__ import annotations

import numpy as np


def parameter_count(n_layers: int) -> int:
    """Return block1D parameter count: (n_layers - 1) + n_layers."""
    if n_layers < 2:
        raise ValueError("n_layers must be >= 2.")
    return 2 * n_layers - 1


def split_model(model: np.ndarray, n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a model vector into thickness and resistivity vectors.

    Parameters
    ----------
    model:
        Physical model vector [thk_1..thk_(n-1), rho_1..rho_n].
    n_layers:
        Number of layers.
    """
    expected = parameter_count(n_layers)
    if model.size != expected:
        raise ValueError(f"Model size {model.size} != expected {expected}.")
    n_thk = n_layers - 1
    return model[:n_thk].copy(), model[n_thk:].copy()


def pack_model(thickness: np.ndarray, resistivity: np.ndarray) -> np.ndarray:
    """Create model vector [thickness, resistivity] from components."""
    if thickness.ndim != 1 or resistivity.ndim != 1:
        raise ValueError("thickness and resistivity must be 1D arrays.")
    if resistivity.size != thickness.size + 1:
        raise ValueError("resistivity size must be thickness size + 1.")
    return np.hstack([thickness, resistivity]).astype(float, copy=False)


def create_start_model(
    data: np.ndarray,
    n_layers: int,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """Create a VES-style start model for block1D inversion.

    This mirrors pyGIMLi's layered start strategy:
    - log-spaced interface depths
    - thicknesses from differences of those depths
    - resistivity initialized with median(data)
    """
    if min_depth <= 0 or max_depth <= 0:
        raise ValueError("min_depth and max_depth must be > 0 for log-spacing.")
    if max_depth <= min_depth:
        raise ValueError("max_depth must be greater than min_depth.")
    depths = np.logspace(np.log10(min_depth), np.log10(max_depth), n_layers - 1)
    thickness = np.diff(np.hstack([[0.0], depths]))
    rho0 = float(np.median(np.asarray(data, dtype=float)))
    resistivity = np.full(n_layers, rho0, dtype=float)
    return pack_model(thickness, resistivity)


def validate_positive_model(model: np.ndarray) -> None:
    """Ensure model values are positive and finite for log transform."""
    if not np.all(np.isfinite(model)):
        raise ValueError("Model contains non-finite values.")
    if np.any(model <= 0):
        raise ValueError("Model values must be > 0 for log-domain inversion.")

