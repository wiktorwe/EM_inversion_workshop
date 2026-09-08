"""Empymod y-integrated magnetic line-source forward (validation-grade fallback).

Synthesises the 2D TE line source by numerically integrating `empymod.dipole`
over the out-of-plane (y) axis: receiver fixed at y=0, Hx dipoles swept along
y'. Matches rockem-suite's native `magnetic_line_source_fields_layered` exactly,
after the source-polarity sign flip derived and measured at `_EMPY_LINE_SIGN`
below (see also `greens_layered_2d_empymod_check.py` in rockem-suite).

The default `n_y=120` quadrature is NOT accurate enough to verify that: it
leaves up to 14 % error at the near offsets. Raise `n_y` and `y_max_m`
(and grade the grid toward y=0) before drawing any convention conclusion from
a comparison against this module.

NOT for production inversion loops — ~35-50x slower than the native kx-domain
solver. Intended only as a fallback when the analytic solver rejects a model
because the source sits on a contrasted interface.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from scripts.modules.analytic_1d_forward import Layer1D, layers_from_rho_thk
from scripts.modules.rockem_bridge import model

# Sign convention, DERIVED and then MEASURED - not an empirical fudge.
#
# Derivation. rockem codes Faraday's law as (WavesEmTE2D::forwardstepHfield /
# insertPhysicalSource)
#       mu dH/dt = -curl E + K        =>   curl E + i*omega*mu*H = +K
# so its magnetic line source K enters with a PLUS sign. empymod follows
# Hunziker et al. (2015), which writes
#       curl E + z_hat*H = -J^m
# i.e. the magnetic source enters with a MINUS sign; empymod's own
# `utils.check_ab` carries the matching relation in-source as
#       G^mm_ab(s, r, e, z) = -G^ee_ab(s, r, -z, -e).
# Hence K = -J^m and the two codes' responses to the same numerical source
# strength differ by exactly the real factor -1.
#
# This is a SOURCE-POLARITY convention, not a time convention. A time-convention
# mismatch (exp(+i w t) vs exp(-i w t)) would show up as complex CONJUGATION,
# which an amplitude-only comparison cannot distinguish from agreement - the
# mistake that has already cost time on this codebase. So the check below was
# done on COMPLEX values.
#
# Measurement (3-layer stack 30/5/80 Ohm-m, tx 100 m, rx 115 m, offsets
# +-13.1/25.3 m, 1 and 4 kHz), taking the raw empymod/native complex ratio with
# this constant set to +1 and a converged y-quadrature (graded grid,
# y_max = 60 km, 6000 points):
#
#       ratio = -1.000011 - 0.000002j  ...  -0.999993 - 0.000005j
#
# i.e. exactly -1 to 1e-5, with zero imaginary part, and FLAT in frequency,
# offset, receiver depth and component (both Hx and Hz). The conjugate ratio is
# nowhere near -1, confirming it is not a time-convention flip. The coarse
# default quadrature (n_y=120) is what made an earlier check look like a "uniform
# pi phase flip with amplitude agreement" rather than an exact -1: at n_y=120 the
# near-offset integrand is under-resolved by up to 14 %.
#
# Independent corroboration that the sign belongs on the EMPYMOD side, not the
# native one: the workshop's FDTD-vs-analytic calibration fits C = FDTD/native
# and gets a phase of +0.02 to -0.12 deg, not 180 deg - so the native solver
# already agrees in polarity with rockem's own engine.
_EMPY_LINE_SIGN = -1.0


class EmpymodUnavailable(RuntimeError):
    """Raised when empymod is required but not installed."""


def _require_empymod():
    try:
        import empymod
    except ImportError as exc:
        raise EmpymodUnavailable(
            "empymod line-source fallback requires empymod (pip install empymod)."
        ) from exc
    return empymod


def empymod_line_yintegral(
    offsets_m: np.ndarray,
    freq_hz: float,
    layers: Sequence[Layer1D],
    tx_depth_m: float,
    rx_depth_m: float,
    ab: int,
    n_y: int = 120,
    y_max_m: float | None = None,
) -> np.ndarray:
    """Line-source response via integral of empymod.dipole(..., ab) over source y'.

    `ab` is empymod's RECEIVER-then-SOURCE code (44 = Hx<-Kx, 64 = Hz<-Kx).
    """
    empymod = _require_empymod()
    depth, res, epermH = model.layers_to_stack(layers, tx_depth_m)
    offsets_m = np.asarray(offsets_m, dtype=float).reshape(-1)
    if y_max_m is None:
        y_max_m = max(5.0 * float(np.max(np.abs(offsets_m))), 200.0)
    y = np.linspace(-y_max_m, y_max_m, int(n_y))
    out = np.zeros(offsets_m.size, dtype=complex)
    for i, off in enumerate(offsets_m):
        resp = empymod.dipole(
            src=[0.0, y, float(tx_depth_m)],
            rec=[float(off), 0.0, float(rx_depth_m)],
            depth=depth,
            res=res,
            freqtime=np.asarray([freq_hz], dtype=float),
            ab=int(ab),
            epermH=epermH,
            epermV=epermH,
            aniso=np.ones(len(res)),
            verb=0,
        )
        out[i] = _EMPY_LINE_SIGN * np.trapezoid(np.asarray(resp, dtype=complex).ravel(), y)
    return out


def forward_empymod_line_gains(
    rho: np.ndarray,
    thickness: np.ndarray,
    freqs_hz: np.ndarray,
    off_x: np.ndarray,
    tx_depth_m: float,
    rx_depth_m: float,
    eps_r: float,
    n_y: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """Complex (Hx, Hz) channel gain per unit Kx source, shape [nfreq, nrx].

    Same I/O convention as `analytic_1d_forward.forward_1d_gains`.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
    off_x = np.asarray(off_x, dtype=float).reshape(-1)
    layers: List[Layer1D] = layers_from_rho_thk(rho, thickness, eps_r)

    nfreq, nrx = freqs_hz.size, off_x.size
    hx = np.full((nfreq, nrx), np.nan, dtype=complex)
    hz = np.full((nfreq, nrx), np.nan, dtype=complex)
    for ifreq, f in enumerate(freqs_hz):
        hx[ifreq, :] = empymod_line_yintegral(
            off_x, float(f), layers, tx_depth_m, rx_depth_m, ab=44, n_y=n_y,
        )
        hz[ifreq, :] = empymod_line_yintegral(
            off_x, float(f), layers, tx_depth_m, rx_depth_m, ab=64, n_y=n_y,
        )

    if not (np.all(np.isfinite(hx)) and np.all(np.isfinite(hz))):
        raise RuntimeError("non-finite empymod line-source forward result")
    return hx, hz


__all__ = [
    "EmpymodUnavailable",
    "empymod_line_yintegral",
    "forward_empymod_line_gains",
]
