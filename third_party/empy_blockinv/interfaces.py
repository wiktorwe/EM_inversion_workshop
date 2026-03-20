"""Public interfaces and configuration objects for block1D inversion.

This module is intentionally verbose and heavily documented so it can serve as
the integration point for projects that provide their own forward modelling and
misfit logic (e.g., Empymod based phase responses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

ArrayLike = np.ndarray


# The forward callback receives a physical model vector:
# [thk_1, ..., thk_(nlay-1), rho_1, ..., rho_n]
# and must return the predicted data vector.
ForwardFunction = Callable[[ArrayLike, Any], ArrayLike]


# Optional callback for custom residual construction.
# If omitted, the package uses (obs - pred) / err.
MisfitFunction = Callable[[ArrayLike, ArrayLike, ArrayLike, Any], ArrayLike]


@dataclass(slots=True)
class InversionConfig:
    """Inversion controls that mirror the pyGIMLi 1D Marquardt workflow."""

    n_layers: int
    start_model: ArrayLike | None = None
    lam: float = 1000.0
    lambda_factor: float = 0.8
    min_lambda: float = 0.0
    max_iter: int = 20
    min_dphi_percent: float = 1.0
    stop_at_chi1: bool = False
    local_regularization: bool = True
    line_search: bool = True
    perturb_factor: float = 1.05
    fd_parallel: bool = False
    fd_workers: int | None = None
    # If True, the damping term is centered at the provided reference model.
    # If False, the damping target is zero in transformed space (log(1)=0).
    use_reference_model: bool = False
    reference_model: ArrayLike | None = None
    # Optional lower/upper parameter limits (physical domain). If used,
    # transformed updates are clipped to these limits after inversion update.
    lower_bound: float | None = None
    upper_bound: float | None = None
    # Optional context passed to callbacks.
    callback_context: Any = None
    # Numerical conditioning parameter added to normal equations.
    ridge_epsilon: float = 1e-12


@dataclass(slots=True)
class InversionState:
    """Container with full inversion run state for diagnostics and plotting."""

    model: ArrayLike
    response: ArrayLike
    residual: ArrayLike
    chi2: float
    phi_data: float
    phi_model: float
    lam: float
    iteration: int
    jacobian_data_model: ArrayLike
    jacobian_residual_model: ArrayLike


@dataclass(slots=True)
class InversionResult:
    """Final result object returned by the inversion runner."""

    model: ArrayLike
    response: ArrayLike
    residual: ArrayLike
    chi2_history: list[float] = field(default_factory=list)
    phi_history: list[float] = field(default_factory=list)
    lambda_history: list[float] = field(default_factory=list)
    model_history: list[ArrayLike] = field(default_factory=list)
    final_state: InversionState | None = None

