"""Differential-evolution smoke test for Empymod layered inversion.

This example mirrors the workflow style from `05_1d_empymod_inversion.ipynb`:
- generate synthetic wrapped phase-only observations with Empymod
- invert a 1D layered model via `scipy.optimize.differential_evolution`
- plot true vs recovered resistivity model (fixed depth span)

The forward engine is the workshop helper:
`[scripts/modules/empymod_1d_forward.py](../../../../scripts/modules/empymod_1d_forward.py)`
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from third_party.empy_blockinv.model import pack_model, split_model

from scripts.modules.empymod_1d_forward import (
    forward_empymod_1d_layered_amp_phase,
    wrap_phase_rad,
)


def _unpack_params(
    params: np.ndarray,
    *,
    n_layers: int,
    span_target: float,
    thk_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map optimizer parameters to physical (rho, thk) with fixed span."""
    p = np.asarray(params, dtype=float)
    if p.size != (2 * n_layers - 1):
        raise ValueError(f"Expected {2 * n_layers - 1} params, got {p.size}.")

    lrho = p[:n_layers]
    lthk = p[n_layers:]

    rho = np.power(10.0, lrho, dtype=float)
    thk_raw = np.power(10.0, lthk, dtype=float)
    thk_raw = np.clip(thk_raw, float(thk_min), np.inf)

    # Enforce fixed span by renormalizing thickness.
    thk_sum = float(np.sum(thk_raw))
    if not np.isfinite(thk_sum) or thk_sum <= 0.0:
        thk_sum = 1e-12
    thk = thk_raw / thk_sum * float(span_target)

    return rho, thk


def _cos_sin_misfit(
    params: np.ndarray,
    *,
    n_layers: int,
    span_target: float,
    thk_min: float,
    freqs: np.ndarray,
    off_x: np.ndarray,
    off_z: np.ndarray,
    tilt_deg: float,
    ab_hxh: int,
    ab_hxhz: int,
    obs_phi_hxh: np.ndarray,
    obs_phi_hxhz: np.ndarray,
    phase_comp_deg: float,
    w_hxh: float,
    w_hxhz: float,
    phase_error_abs: float,
) -> float:
    """Unit-circle objective (exp(i*phi)) using cosine/sine equivalence.

    Despite the function name, this implements the original unit-circle
    complex-exponential misfit:
      pred = exp(i*phi_pred)
      obs  = exp(i*phi_obs_comp)
      mis = sum(|pred-obs|^2)
    which is unique and wrap-robust for phase.
    """
    rho, thk = _unpack_params(
        params,
        n_layers=n_layers,
        span_target=span_target,
        thk_min=thk_min,
    )

    _amp_hxh, phi_hxh, _amp_hxhz, phi_hxhz = forward_empymod_1d_layered_amp_phase(
        rho=rho,
        thickness=thk,
        freqs=freqs,
        off_x=off_x,
        off_z=off_z,
        tilt_deg=tilt_deg,
        ab_hxh=ab_hxh,
        ab_hxhz=ab_hxhz,
        verb=0,
    )

    # Apply optional systematic phase compensation.
    shift = float(phase_comp_deg)
    obs_hxh_c = wrap_phase_rad(obs_phi_hxh - np.deg2rad(shift))
    obs_hxhz_c = wrap_phase_rad(obs_phi_hxhz - np.deg2rad(shift))

    phi_hxh = np.asarray(phi_hxh, dtype=float)
    phi_hxhz = np.asarray(phi_hxhz, dtype=float)

    # Unit-circle residual: complex exponential.
    pred1 = np.exp(1j * phi_hxh)
    pred2 = np.exp(1j * phi_hxhz)
    obs1 = np.exp(1j * obs_hxh_c)
    obs2 = np.exp(1j * obs_hxhz_c)

    res1 = pred1 - obs1
    res2 = pred2 - obs2

    mis = 0.0
    if np.any(np.isfinite(res1)):
        mis += float(w_hxh) * float(np.nansum(np.abs(res1) ** 2))
    if np.any(np.isfinite(res2)):
        mis += float(w_hxhz) * float(np.nansum(np.abs(res2) ** 2))
    return float(mis)


def main() -> None:
    try:
        from scipy.optimize import differential_evolution
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'scipy'. Install with: pip install scipy"
        ) from exc

    import matplotlib.pyplot as plt

    # -------------------- Problem setup --------------------
    n_layers = 4
    true_thk = np.array([5.0, 10.0, 20.0], dtype=float)
    true_rho = np.array([100.0, 20.0, 200.0, 50.0], dtype=float)
    span_target = float(np.sum(true_thk))

    freqs = np.asarray([2000.0, 4000.0, 8000.0], dtype=float)
    # Keep geometry compact for quick runs.
    off_x = np.array([-13.1, -25.3])
    off_z = np.ones_like(off_x, dtype=float) * span_target/2
    tilt_deg = 0.5

    ab_hxh = 44
    ab_hxhz = 46

    # Bounds in log10-space (chosen for a smoke test).
    log10_rho_min = np.log10(1.0)
    log10_rho_max = np.log10(500.0)
    thk_min = 1.0  # meters (physical)
    thk_max = span_target  # meters (physical, but span is fixed anyway)
    log10_thk_min = np.log10(thk_min)
    log10_thk_max = np.log10(thk_max)

    # -------------------- Synthetic observations --------------------
    _amp_hxh, phi_hxh_true, _amp_hxhz, phi_hxhz_true = forward_empymod_1d_layered_amp_phase(
        rho=true_rho,
        thickness=true_thk,
        freqs=freqs,
        off_x=off_x,
        off_z=off_z,
        tilt_deg=tilt_deg,
        ab_hxh=ab_hxh,
        ab_hxhz=ab_hxhz,
        verb=0,
    )

    rng = np.random.default_rng(1337)
    phase_error_abs = 0.03  # rad
    obs_phi_hxh = wrap_phase_rad(phi_hxh_true + rng.normal(scale=phase_error_abs, size=phi_hxh_true.shape))
    obs_phi_hxhz = wrap_phase_rad(phi_hxhz_true + rng.normal(scale=phase_error_abs, size=phi_hxhz_true.shape))

    # -------------------- Differential evolution --------------------
    # Parameter vector: [log10(rho_1..rho_n), log10(thk_1..thk_(n-1))]
    bounds = []
    for _ in range(n_layers):
        bounds.append((log10_rho_min, log10_rho_max))
    for _ in range(n_layers - 1):
        bounds.append((log10_thk_min, log10_thk_max))

    phase_comp_deg = 0.0
    w_hxh = 1.0
    w_hxhz = 1.0

    obj = lambda p: _cos_sin_misfit(
        p,
        n_layers=n_layers,
        span_target=span_target,
        thk_min=thk_min,
        freqs=freqs,
        off_x=off_x,
        off_z=off_z,
        tilt_deg=tilt_deg,
        ab_hxh=ab_hxh,
        ab_hxhz=ab_hxhz,
        obs_phi_hxh=obs_phi_hxh,
        obs_phi_hxhz=obs_phi_hxhz,
        phase_comp_deg=phase_comp_deg,
        w_hxh=w_hxh,
        w_hxhz=w_hxhz,
        phase_error_abs=phase_error_abs,
    )

    result = differential_evolution(
        obj,
        bounds=bounds,
        maxiter=30,
        popsize=20,
        seed=40,
        polish=False,
        workers=1,
        updating="deferred",
    )

    best_params = np.asarray(result.x, dtype=float)
    best_rho, best_thk = _unpack_params(
        best_params,
        n_layers=n_layers,
        span_target=span_target,
        thk_min=thk_min,
    )

    print("DE success:", bool(getattr(result, "success", True)))
    print("DE fun (misfit):", float(result.fun))
    print("True rho:", true_rho)
    print("Best rho:", best_rho)
    print("True thk:", true_thk, "sum(thk)=", float(np.sum(true_thk)))
    print("Best thk:", best_thk, "sum(thk)=", float(np.sum(best_thk)))

    # -------------------- Plot true vs inverted --------------------
    from third_party.empy_blockinv.plotting import plot_block_model

    true_model = pack_model(true_thk, true_rho)
    inv_model = pack_model(best_thk, best_rho)

    _, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

    # Plot the final DE population in light gray to visualize the search.
    population = getattr(result, "population", None)
    population_energies = getattr(result, "population_energies", None)
    if population is not None and np.size(population) > 0:
        population = np.asarray(population, dtype=float)
        n_pop = int(population.shape[0])

        if population_energies is not None and np.size(population_energies) == n_pop:
            order = np.argsort(np.asarray(population_energies, dtype=float))
        else:
            order = np.arange(n_pop, dtype=int)

        # Plot all if reasonable; otherwise limit for performance/readability.
        max_pop_to_plot = min(n_pop, 200)
        idx_to_plot = order[:max_pop_to_plot]
        denom = max(1, len(idx_to_plot) - 1)

        for j, ip in enumerate(idx_to_plot):
            params_i = population[ip]
            rho_i, thk_i = _unpack_params(
                params_i,
                n_layers=n_layers,
                span_target=span_target,
                thk_min=thk_min,
            )
            model_i = pack_model(thk_i, rho_i)
            gray = 0.85 - 0.65 * (j / denom)  # darker for better candidates
            color = (gray, gray, gray, 0.10)
            plot_block_model(model_i, n_layers=n_layers, ax=ax, label="_nolegend_", color=color)

    # Overlay best and true on top.
    plot_block_model(inv_model, n_layers=n_layers, ax=ax, label="inverted", color="C1")
    plot_block_model(true_model, n_layers=n_layers, ax=ax, label="true", color="C0")
    ax.set_title("Differential Evolution: true vs recovered (fixed span)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()

