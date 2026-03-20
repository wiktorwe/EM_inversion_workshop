"""Line-search implementation mirroring pyGIMLi's RInversion behavior."""

from __future__ import annotations

from typing import Callable

import numpy as np


def quadratic_tau(
    phi0: float,
    phi1: float,
    phit: float,
    tauquad: float,
) -> float:
    """Parabolic step-length estimate from three objective evaluations."""
    phi10 = phi1 - phi0
    phit0 = phit - phi0
    dphit = phit0 - phi10 * tauquad
    if abs(dphit) < 1e-15:
        return 0.0
    return (phit0 - phi10 * tauquad * tauquad) / dphit / 2.0


def line_search_tau(
    objective_for_tau: Callable[[float], float],
    local_regularization: bool = True,
    tauquad: float = 0.3,
) -> float:
    """Search tau in [0.01, 1.0] with pyGIMLi-style fallback.

    Parameters
    ----------
    objective_for_tau:
        Callable returning objective value at a given tau.
    local_regularization:
        Kept for API parity and explicit readability.
    tauquad:
        Secondary trial step for parabolic fallback.
    """
    # pyGIMLi scans 100 candidate taus with increment 0.01.
    taus = np.arange(0.01, 1.01, 0.01)
    vals = np.array([objective_for_tau(tau) for tau in taus], dtype=float)
    best_idx = int(np.argmin(vals))
    tau = float(taus[best_idx])

    if tau < 0.03:
        phi0 = float(objective_for_tau(0.0))
        phi1 = float(objective_for_tau(1.0))
        phit = float(objective_for_tau(tauquad))
        tau = quadratic_tau(phi0=phi0, phi1=phi1, phit=phit, tauquad=tauquad)
        if tau > 1.0:
            tau = 1.0
        if tau < 0.03:
            tau = 0.03

    return float(tau)

