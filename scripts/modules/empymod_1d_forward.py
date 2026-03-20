"""Empymod 1D layered forward for inversion results.

This helper is used by `06_1d_empymod_results.ipynb` to recompute synthetic
amp/phase (Hx and Hz) from the inverted layered resistivity model.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def wrap_phase_rad(phi: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi] (stable for arrays)."""
    phi = np.asarray(phi, dtype=float)
    return np.arctan2(np.sin(phi), np.cos(phi))


def ab_with_z_oriented_source(ab_code: int) -> int:
    """Map empirical component code to the z-oriented source counterpart."""
    ab = int(ab_code)
    src_code = ab // 10
    rec_code = ab % 10
    # In this workshop's empymod encoding, (+2) changes x->z within E/H families.
    return (src_code + 2) * 10 + rec_code


def forward_empymod_1d_layered_amp_phase(
    *,
    rho: np.ndarray,
    thickness: np.ndarray,
    freqs: np.ndarray,
    off_x: np.ndarray,
    off_z: np.ndarray,
    tilt_deg: float,
    ab_hxh: int = 44,
    ab_hxhz: int = 46,
    verb: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward model a layered 1D earth and return amp/phase for Hx and Hz.

    Parameters
    ----------
    rho:
        Layer resistivities (length n_layers).
    thickness:
        Layer thicknesses in log10-space during inversion; but here they are the
        *resulting* thickness values. Length must be n_layers - 1.
    freqs:
        Frequencies (Hz), shape [nfreq].
    off_x/off_z:
        Receiver offsets relative to the transmitter location, shape [nrx].
    tilt_deg:
        Optional source tilt angle mixing x- and z-oriented components.

    Returns
    -------
    amp_hxh, phi_hxh, amp_hxhz, phi_hxhz:
        Arrays of shape [nfreq, nrx].
    """

    try:
        import empymod  # local import so notebooks can fail with a good message
    except Exception as exc:
        raise RuntimeError("Missing dependency 'empymod'. Install with: pip install empymod") from exc

    rho = np.asarray(rho, dtype=float)
    thickness = np.asarray(thickness, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    off_x = np.asarray(off_x, dtype=float)
    off_z = np.asarray(off_z, dtype=float)

    n_layers = int(rho.size)
    if n_layers < 1:
        raise ValueError("rho must contain at least one layer.")

    thickness = thickness.reshape(-1)
    if n_layers > 1:
        if int(thickness.size) != n_layers - 1:
            raise ValueError(
                f"Thickness length mismatch: expected {n_layers - 1}, got {thickness.size}"
            )
    else:
        if thickness.size not in (0, 1):
            raise ValueError("For n_layers=1, thickness must be empty or length 1.")
        thickness = np.array([], dtype=float)

    nfreq = int(freqs.size)
    nrx = int(off_x.size)

    # In `05_1d_empymod_inversion.ipynb`, z_start_rel/z_end_rel are centered around
    # the transmitter depth: z_start_rel = -halfspan, z_end_rel = +halfspan.
    # The span is the sum of thickness values (because inversion normalizes thicknesses to span).
    span = float(np.sum(thickness)) if thickness.size else 0.0
    if span <= 0.0:
        # Guard for unexpected single-layer or degenerate thickness.
        span = 1.0

    z_start_rel = -0.5 * span
    depth_rel = z_start_rel + np.cumsum(thickness, dtype=float) if thickness.size else np.array([], dtype=float)

    # dipole() expects src=[x, y, z]. We solve in a tx-centered coordinate system.
    src = [0.0, 0.0, 0.0]
    common = {
        "src": src,
        "depth": depth_rel,
        "res": rho,
        "freqtime": freqs,
        "verb": int(verb),
        "htarg": {"pts_per_dec": -1},
    }

    def _forward_component(ab_code: int) -> np.ndarray:
        """Return complex response for a single empymod component."""
        if nrx == 0:
            return np.zeros((nfreq, 0), dtype=complex)

        # empymod.dipole vectorization works if off_z is constant across receivers.
        if np.allclose(off_z, off_z[0], rtol=0.0, atol=1e-9):
            rec_vec = [off_x, 0.0, float(off_z[0])]
            resp = empymod.dipole(ab=int(ab_code), rec=rec_vec, **common)
            # Ensure shape = (nfreq, nrx)
            resp = np.asarray(resp, dtype=complex)
            if resp.ndim == 1:
                return np.repeat(resp[:, None], nrx, axis=1)
            if resp.shape == (nfreq, nrx):
                return resp
            if resp.shape == (nrx, nfreq):
                return resp.T
            return np.reshape(resp, (nfreq, nrx))

        out = np.full((nfreq, nrx), np.nan + 1j * np.nan, dtype=complex)
        for ir in range(nrx):
            rec_i = [float(off_x[ir]), 0.0, float(off_z[ir])]
            resp_i = empymod.dipole(ab=int(ab_code), rec=rec_i, **common)
            resp_i = np.asarray(resp_i, dtype=complex)
            if resp_i.ndim == 0:
                out[:, ir] = resp_i
            elif resp_i.ndim == 1:
                out[:, ir] = resp_i
            else:
                out[:, ir] = np.reshape(resp_i, (nfreq,))
        return out

    # Base responses: x-oriented source component and x->z tilted alternative.
    hxh = _forward_component(ab_hxh)
    hxhz = _forward_component(ab_hxhz)

    # Optional tilt mixing in x-z plane by combining x- and z-oriented source components.
    theta = np.deg2rad(float(tilt_deg))
    if abs(theta) > 0.0:
        ab_hxh_z = ab_with_z_oriented_source(ab_hxh)
        ab_hxhz_z = ab_with_z_oriented_source(ab_hxhz)
        hxh_z = _forward_component(ab_hxh_z)
        hxhz_z = _forward_component(ab_hxhz_z)
        hxh = np.cos(theta) * hxh + np.sin(theta) * hxh_z
        hxhz = np.cos(theta) * hxhz + np.sin(theta) * hxhz_z

    amp_hxh = np.abs(hxh)
    amp_hxhz = np.abs(hxhz)
    phi_hxh = wrap_phase_rad(np.angle(hxh))
    phi_hxhz = wrap_phase_rad(np.angle(hxhz))

    return amp_hxh, phi_hxh, amp_hxhz, phi_hxhz


__all__ = [
    "wrap_phase_rad",
    "ab_with_z_oriented_source",
    "forward_empymod_1d_layered_amp_phase",
]

