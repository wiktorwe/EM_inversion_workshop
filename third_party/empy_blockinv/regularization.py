"""Regularization helpers matching pyGIMLi cType=0 damping behavior."""

from __future__ import annotations

import numpy as np


def ctype0_constraints(n_params: int) -> np.ndarray:
    """Return identity constraints matrix for cType=0 style damping."""
    return np.eye(n_params, dtype=float)


def damping_target(
    model_transformed: np.ndarray,
    use_reference_model: bool,
    reference_transformed: np.ndarray | None = None,
) -> np.ndarray:
    """Return damping target in transformed model space.

    - Without reference model: target is zero (equivalent to log(1)=0).
    - With reference model: target is transformed reference.
    """
    if use_reference_model:
        if reference_transformed is None:
            raise ValueError("reference_transformed must be provided when enabled.")
        return np.asarray(reference_transformed, dtype=float)
    return np.zeros_like(np.asarray(model_transformed, dtype=float))


def phi_model_c0(
    model_transformed: np.ndarray,
    use_reference_model: bool,
    reference_transformed: np.ndarray | None = None,
) -> float:
    """Compute cType=0 model objective in transformed space."""
    target = damping_target(
        model_transformed=model_transformed,
        use_reference_model=use_reference_model,
        reference_transformed=reference_transformed,
    )
    dm = np.asarray(model_transformed, dtype=float) - target
    return float(dm @ dm)

