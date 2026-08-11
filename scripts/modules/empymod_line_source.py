"""Empymod y-integrated magnetic line-source forward (validation-grade fallback).

Synthesises the 2D TE line source by numerically integrating `empymod.dipole`
over the out-of-plane (y) axis: receiver fixed at y=0, Hx dipoles swept along
y'. Matches rockem-suite's native `magnetic_line_source_fields_layered` after
a uniform sign flip (see `greens_layered_2d_empymod_check.py` in rockem-suite).

NOT for production inversion loops — ~35-50x slower than the native kx-domain
solver. Intended only as a fallback when the analytic solver rejects a model
because the source sits on a contrasted interface.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from scripts.modules.analytic_1d_forward import Layer1D, layers_from_rho_thk
from scripts.modules.rockem_bridge import model

# empymod ab=44/64 y-integrated line response matches the native 2D Kx line-
# source solver in amplitude but with a uniform pi phase flip.
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
