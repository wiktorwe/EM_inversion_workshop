"""The FDTD-vs-analytic discrepancy, computed rather than fitted.

WHY THIS EXISTS
---------------
The obvious alternative to computing `C(f)` is to FIT it: run an extra FDTD job
on a reference Earth and least-squares match it to the analytic solver
(`fdtd_analytic_calibration.fit_global_C_per_frequency`). Two things are wrong
with that as a PRODUCTION mechanism:

1. **It measures almost nothing.** `|C|` is derivable. The engine injects the
   wavelet into one cell as ``H += (dt/MU) * wav`` (`WavesEmTE2D.cpp:796,817`),
   which represents ``K * delta`` integrated over that cell - so the effective
   moment is ``wav * dx * dz`` and ``|C| = dx*dz``. Measured at order 6:
   ``|C|/dx^2 = 0.99931 - 0.99954``. The remaining ~0.05 % is the stencil
   consistency factor computed below (0.012 %) plus an offset-dependent term.

2. **It is fitted where a component is a near-null.** For a Kx line source in a
   layered Earth, Hz vanishes identically at the source depth. The homogeneous
   calibration breaks that with receivers at +/-20 m; the lateral-average one
   keeps the production colinear geometry, so Hz is fitted ON the null. That is
   where its 2.6-6.5 % residual scatter comes from - and that scatter becomes
   `sigma` for `Cxz`/`Czz`, the cross-couplings that carry the look-ahead
   signal. A noise estimate manufactured by a bad geometry is worse than no
   calibration at all.

So `C` is computed here, and the FDTD calibration run is a VALIDATION that the
computed value holds rather than the source of it.

WHAT IS AND IS NOT IN THE BUDGET
--------------------------------
Terms, with the measurement or derivation behind each:

* ``dx*dz``            source-injection cell moment. Exact, derived, and checked
                       three ways: Kx vs Kz agree to 0.02 %, the homogeneous
                       whole-space case, and a historical 2x that was traced to
                       an on-interface source rather than an engine bug.
* ``s = sum c_n(2n+1)`` stencil consistency. The tabulated coefficients are
                       dispersion-optimised, not Taylor-exact, so every first
                       derivative is scaled by `s`. 0.999882 at order 6;
                       0.977010 at order 2, where it is a 2.3 % error that does
                       NOT converge under refinement.
* ``(dx/r)^2``         the source AND receiver bilinear 2x2 kernels are
                       finite-support, not deltas. Measured 1.3 % at 7.9 cells
                       against 1.6 % predicted; converges as dx^2.
* grid dispersion      ``D(k)/k - 1`` from the exact discrete symbol. <=0.01 %
                       at order 6 over 8-40 points per wavelength - negligible
                       there, and the reason order 6 was chosen.
* interface quantisation  half a cell of interface placement. NOT closed form -
                       measured per frequency and per receiver component.
* kx quadrature        the ANALYTIC side's own error, 0.22 % worst case.
* ``eps_r`` inflation  ZERO in this budget. It biases both sides equally
                       because the analytic reference is evaluated at the same
                       `eps_r_used`, so it cancels out of the discrepancy. It is
                       a real 0.15-0.26 % bias against TRUE physics; that is a
                       different budget.
* PML                  <1e-4 %. Not in the budget.

Nothing here runs an FDTD job or reads a workspace.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

# The analytic solver's own kx-quadrature error at the shipped policy
# (LAM_MAX_MULTIPLIER = 2.0, n_nodes = 120), measured over the workshop's prior
# by `scripts/experiments/kx_convergence.py`. It is an error in the PREDICTION,
# not in the data, but it enters the residual identically.
QUADRATURE_REL_ERROR = 0.0022

# Floor on the relative budget. Nothing is known to better than this, and it
# keeps a near-null datum from being assigned a near-zero sigma - which is the
# failure mode of scaling sigma by |obs|.
MIN_REL_ERROR = 1e-3


def _stencil_coeffs(order: int) -> Sequence[float]:
    """The staggered first-derivative coefficient row for `order`.

    Read from rockem-suite, never re-tabulated here. They already exist twice
    (`lib/der/der.cpp`'s w** table and `python/rockem/utils/utils.py`'s mirror
    of it); a third copy in this repo is exactly the drift RULE 4 is about.
    """
    from scripts.modules.rockem_bridge import utils as _rockem_utils  # noqa: F401
    from rockem.utils.utils import _STENCIL_COEFFS

    return _STENCIL_COEFFS[max(1, min(8, int(order)))]


def stencil_consistency(order: int) -> float:
    """`s = sum_n c_n (2n+1)`, the scale every first derivative picks up.

    A staggered first derivative is exact for a linear field only if
    `sum_n c_n (2n+1) == 1`. The tabulated coefficients are dispersion-optimised
    instead, so `s != 1` and BOTH curls carry the factor - the discrete
    curl-curl is scaled by `s^2` and the numerical skin depth becomes `s*delta`.
    That error is independent of `dx`, so it does NOT converge under refinement;
    it is why the workshop runs order 6 and not order 2.

    Reproduces `scripts/templates/mod.cfg`'s table exactly:
    order 2 -> 0.977010, 4 -> 0.998488, 6 -> 0.999882, 8 -> 0.999990.
    """
    row = _stencil_coeffs(order)
    return float(sum(c * (2 * (n + 1) - 1) for n, c in enumerate(row)))


def derivative_symbol_error(order: int, k_dx) -> np.ndarray:
    """`D(k)/k - 1` for the discrete staggered first derivative.

    `D(k) = (2/dx) * sum_m w_m sin((2m-1) k dx / 2)` is the exact symbol. It is
    already written down in rockem-suite, but only ever evaluated at the
    grid-Nyquist mode `k dx = pi` to get the CFL factor kappa; evaluating it at
    the wavenumbers the survey actually contains is what turns it into an error
    estimate. `k_dx -> 0` recovers `stencil_consistency(order) - 1`.
    """
    row = _stencil_coeffs(order)
    kdx = np.asarray(k_dx, dtype=float)
    small = np.abs(kdx) < 1e-12
    safe = np.where(small, 1.0, kdx)
    d_over_k = sum(w * np.sin((2 * (m + 1) - 1) * safe / 2.0)
                   for m, w in enumerate(row)) * 2.0 / safe
    return np.where(small, stencil_consistency(order) - 1.0, d_over_k - 1.0)


def analytic_C(dx_m: float, order: int, dz_m: Optional[float] = None) -> float:
    """The FDTD-to-analytic scale, in closed form: `dx * dz * s(order)`.

    REAL and positive. The fitted `C` carries a phase of +0.006 to +0.081 deg,
    i.e. 1.4e-3 rad at worst - below every term in the budget, and consistent
    with zero. If a validation run ever reports a phase that is NOT small, that
    is a finding, not something to absorb.
    """
    dz = float(dx_m if dz_m is None else dz_m)
    return float(dx_m) * dz * stencil_consistency(order)


def geometric_rel_error(offsets_m, dx_m: float) -> np.ndarray:
    """`(dx/r)^2` - the finite-support source and receiver kernels.

    Both `insertPhysicalSource` and `recordData` spread over a 2x2 bilinear
    stencil whose weights sum to 1 (so the moment is exact) but whose second
    moment is not zero. The residual behaves as `(dx/r)^2`: measured 1.3 % at
    7.9 cells of offset against `(1/7.9)^2 = 1.6 %` predicted, and it converges
    as `dx^2` (0.263 % -> 0.090 % when dx halves), unlike the stencil term.
    """
    r = np.abs(np.asarray(offsets_m, dtype=float))
    return (float(dx_m) / np.maximum(r, float(dx_m))) ** 2


def interface_quantisation_rel(
    freqs_hz,
    offsets_m,
    tx_depth_m,
    eps_r,
    dx_m: float,
    rho=(2.0, 25.0, 100.0),
    thickness=(40.0, 35.0),
    rx_depth_m=None,
):
    """Relative data change from moving an interface HALF A CELL, per frequency.

    This is the term that stops the inversion resolving below the grid. It is
    MEASURED, not derived: there is no closed form, because it depends on where
    each interface happens to fall and on the contrast across it. What makes it
    a legitimate budget entry rather than a fudge is that half a cell is the
    irreducible placement uncertainty of ANY gridded method - the FD model
    cannot put an interface anywhere else, so the data cannot distinguish
    anything finer.

    Returns `{"HX": [nfreq], "HZ": [nfreq]}` keyed by RECEIVER component.

    The default `rho`/`thickness` is the representative 2/25/100 Ohm-m stack
    `scripts/experiments/interface_snapping.py` reports against, with the
    transmitter inside the middle layer as in the real survey. It is deliberately
    a FIXED reference model, not the candidate being fitted: a sigma that moved
    with the model would make the objective ill-posed.
    """
    from scripts.modules.analytic_1d_forward import forward_1d_gains

    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    off = np.asarray(offsets_m, dtype=float).reshape(-1)
    rx_z = float(tx_depth_m) if rx_depth_m is None else rx_depth_m
    thk = np.asarray(thickness, dtype=float)

    ref_hx, ref_hz = forward_1d_gains(np.asarray(rho, dtype=float), thk, freqs,
                                      off, float(tx_depth_m), rx_z, eps_r)
    moved = thk.copy()
    moved[0] += 0.5 * float(dx_m)
    hx, hz = forward_1d_gains(np.asarray(rho, dtype=float), moved, freqs,
                              off, float(tx_depth_m), rx_z, eps_r)

    def _rel(a, b):
        return np.max(np.abs(np.abs(a) / np.maximum(np.abs(b), 1e-300) - 1.0), axis=1)

    return {"HX": _rel(hx, ref_hx), "HZ": _rel(hz, ref_hz)}


def sigma_budget(
    obs,
    offsets_m,
    dx_m: float,
    order: int,
    quantisation_rel=0.0,
    quadrature_rel: float = QUADRATURE_REL_ERROR,
    min_rel: float = MIN_REL_ERROR,
) -> Dict[str, np.ndarray]:
    """Per-frequency, PER-RECEIVER sigma for one tensor component.

    Returns `{"sigma": [nfreq, nrx], "rel": [nfreq, nrx], "terms": {...}}`.

    `sigma` is RELATIVE to the datum and then scaled by `|obs|`, which is the
    substantive change from the fitted sigma: that one was a single absolute
    number per frequency, inherited from the CALIBRATION run's amplitude scale
    and constant across offsets, so the near offset dominated the misfit and the
    far offset was effectively ignored.

    Scaling by `|obs|` puts noise in the denominator, which is a mild bias. The
    alternative - scaling by `|prediction|` - makes sigma depend on the model
    being fitted and the objective ill-posed, which is worse. `min_rel` and the
    per-frequency median floor below keep a near-null datum from being handed a
    near-zero sigma.

    `quantisation_rel` is the half-cell interface sensitivity for THIS receiver
    component, per frequency. It is measured, not derived - see
    `scripts/experiments/interface_snapping.py` - and it is the term that stops
    the inversion resolving below the grid.

    PASS ZERO FOR IT WHEN THE CANDIDATE IS SNAPPED. Snapping makes the model
    reproduce each grid's quantisation, so the error is already removed from the
    residual; charging for it again in sigma double-counts, and measurably makes
    the recovered model worse (rms log10 rho 0.9357 with both against 0.8417
    with snapping alone, 3 Tx, co-components). The two mechanisms are
    alternatives, not partners.
    """
    a = np.abs(np.asarray(obs, dtype=complex))
    nfreq, nrx = a.shape
    geom = np.broadcast_to(geometric_rel_error(offsets_m, dx_m)[None, :], (nfreq, nrx))
    stencil = abs(stencil_consistency(order) - 1.0)
    quant = np.asarray(quantisation_rel, dtype=float).reshape(-1)
    if quant.size == 1:
        quant = np.full(nfreq, float(quant[0]))
    if quant.size != nfreq:
        raise ValueError(
            f"quantisation_rel has {quant.size} entries but there are {nfreq} frequencies")
    quant2d = np.broadcast_to(quant[:, None], (nfreq, nrx))

    rel = np.sqrt(geom ** 2 + stencil ** 2 + quant2d ** 2 + float(quadrature_rel) ** 2)
    rel = np.maximum(rel, float(min_rel))

    # A near-null datum must not buy itself a tiny sigma. Floor the ABSOLUTE
    # sigma at the same relative error applied to that frequency's median
    # amplitude, so a component that is small everywhere keeps a sane scale and
    # one that is small only at a null offset does not dominate the misfit.
    scale = np.where(a > 0.0, a, np.nan)
    med = np.nanmedian(scale, axis=1)
    med = np.where(np.isfinite(med), med, 1.0)[:, None]
    sigma = np.maximum(rel * a, rel * med * 1e-2)

    return {
        "sigma": sigma,
        "rel": rel,
        "terms": {
            "geometric": geom,
            "stencil": float(stencil),
            "quantisation": quant,
            "quadrature": float(quadrature_rel),
        },
    }


__all__ = [
    "MIN_REL_ERROR",
    "QUADRATURE_REL_ERROR",
    "analytic_C",
    "derivative_symbol_error",
    "geometric_rel_error",
    "sigma_budget",
    "stencil_consistency",
]
