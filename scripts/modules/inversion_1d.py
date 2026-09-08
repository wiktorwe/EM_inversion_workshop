"""Shared 1D layered-inversion core (model parameterisation + misfit).

These four functions were defined inside `05_1d_inversion.ipynb`'s single code
cell, which made them unreachable from any script. They are moved here verbatim
(bar the frequency-subset support added for the multi-scale ladder) so notebook
05 and `scripts/experiments/multiscale_1d.py` share ONE implementation - a
second copy of the misfit is exactly how a prototype and the thing it is
prototyping quietly diverge.

The notebook imports these; it no longer defines them.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from scripts.modules.analytic_1d_forward import ForwardRejected, forward_1d_gains


def unpack_model_params(params, n_layers, z_start_rel, z_end_rel, snap_dz=None,
                        snap_origin_rel=0.0):
    """(rho, thickness, interface depths) from the optimiser's parameter vector.

    Resistivities are optimised in log10 space; thicknesses too, then rescaled
    so they exactly fill the `[z_start_rel, z_end_rel]` window - so the depth
    window is a hard constraint rather than something the optimiser can drift
    out of.

    `snap_dz` snaps the resulting interface depths onto a grid of that spacing
    (relative to `snap_origin_rel`). This exists because the inversion fits a
    CONTINUOUS-interface analytic forward against FDTD data whose interfaces are
    quantised onto the FD grid, a systematic depth error of up to half a cell
    that `C(f)` cannot absorb (C is one complex number per frequency, shared by
    every transmitter; this error is model- and depth-dependent). Snapping puts
    the candidate model on the same footing as the data.

    Default is `None` (no snapping), because whether it is worth doing depends
    on whether half a cell moves the data by more than the noise floor - measure
    it with `scripts/experiments/interface_snapping.py` on your own geometry
    before switching it on, and re-fit the calibration afterwards.

    Snapping is applied to the interface DEPTHS and the thicknesses are then
    recomputed from them, so the two stay consistent; the window endpoints are
    preserved, and interfaces are kept strictly increasing (a snap that would
    collapse two interfaces onto the same cell face is nudged one cell apart
    rather than producing a zero-thickness layer the solver would reject).
    """
    p = np.asarray(params, dtype=float)
    lrho = p[:n_layers]
    rho = np.power(10.0, lrho)

    # Thickness parameters are optimized in log10-space.
    lthk = p[n_layers:]
    thk_raw = np.power(10.0, lthk)
    thk_raw = np.clip(thk_raw, 1e-9, np.inf)

    span = float(z_end_rel - z_start_rel)
    if span <= 0:
        raise ValueError("Invalid depth window: z_end_rel must be > z_start_rel")

    if thk_raw.size != max(0, n_layers - 1):
        raise ValueError("Thickness parameter count mismatch.")

    if thk_raw.size > 0:
        thk = thk_raw / np.sum(thk_raw) * span
        depth = float(z_start_rel) + np.cumsum(thk)
        if snap_dz is not None and float(snap_dz) > 0.0:
            d = float(snap_dz)
            snapped = np.round((depth - float(snap_origin_rel)) / d) * d + float(snap_origin_rel)
            # keep strictly increasing and inside the window
            lo, hi = float(z_start_rel), float(z_end_rel)
            for i in range(snapped.size):
                floor_i = lo + (i + 1) * d
                if i > 0:
                    floor_i = max(floor_i, snapped[i - 1] + d)
                snapped[i] = min(max(snapped[i], floor_i), hi - (snapped.size - i) * d)
            depth = snapped
            thk = np.diff(np.concatenate([[lo], depth]))
    else:
        thk = np.array([], dtype=float)
        depth = np.array([], dtype=float)
    return rho, thk, depth


def forward_analytic_for_tx(params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
                            n_nodes=120, freq_mask=None, snap_dz=None):
    """Analytic (Hx, Hz) complex channel gain for a candidate model, via
    `analytic_1d_forward.forward_1d_gains` - the validated 2D magnetic
    line-source solver, exact counterpart of the FDTD survey (see that
    module's docstring). Raises `ForwardRejected` on an unevaluable model -
    callers must catch it (never let it silently become NaN).

    `freq_mask` restricts the forward to a subset of `tx_entry["freqs"]`. That
    is what makes one stage of the multi-scale ladder a SLICE rather than a
    rewrite: nothing else about the misfit changes between stages.
    """
    rho, thk, _ = unpack_model_params(params, n_layers, z_start_rel, z_end_rel,
                                      snap_dz=snap_dz)
    tx_z = float(tx_entry["tx_z"])
    off_x = np.asarray(tx_entry["off_x"], dtype=float)
    # PER-RECEIVER depths. This used to be `tx_z + off_z[0]`, which silently
    # forced every receiver of a transmitter to the FIRST receiver's depth.
    # Harmless for the workshop's default colinear, zero-depth-offset survey
    # (all off_z are 0), wrong the moment anyone uses depth-offset receivers -
    # and inconsistent with `fdtd_analytic_calibration`'s
    # `layered_analytic_gains`/`homogeneous_analytic_gains`, which have always
    # looped over unique receiver depths. `forward_1d_gains` now accepts the
    # array and does the same grouping.
    rx_depth_m = tx_z + np.asarray(tx_entry["off_z"], dtype=float)
    freqs = np.asarray(tx_entry["freqs"], dtype=float)
    if freq_mask is not None:
        freqs = freqs[np.asarray(freq_mask, dtype=bool)]
    return forward_1d_gains(rho, thk, freqs, off_x, tx_z, rx_depth_m, eps_r, n_nodes=n_nodes)


def complex_gain_objective(
    params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
    reg_lambda, w_hxh, w_hxhz, sigma_hx, sigma_hz, C=None, freq_mask=None, snap_dz=None,
):
    """Complex-gain misfit: ((C*pred - obs)/sigma) summed in quadrature over
    frequency and receiver, for both Hx and Hz, plus a Tikhonov first-
    difference penalty on log10(rho).

    `C` is the active global FDTD-analytic calibration from notebook 02
    (homogeneous or lateral-average), shared by all Tx and both Hx/Hz.

    `freq_mask` (boolean, length nfreq) restricts the misfit to a subset of the
    frequencies - one stage of a multi-scale ladder. `sigma_hx`, `sigma_hz` and
    `C` are per-frequency arrays and are sliced with the same mask, so each
    stage automatically gets that frequency's OWN calibration scatter rather
    than a band-averaged one.
    """
    try:
        hx_pred, hz_pred = forward_analytic_for_tx(
            params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
            freq_mask=freq_mask, snap_dz=snap_dz)
    except ForwardRejected:
        return 1e12

    obs_hx = np.asarray(tx_entry["obs_hx_gain"], dtype=complex)
    obs_hz = np.asarray(tx_entry["obs_hz_gain"], dtype=complex)
    sigma_hx = np.asarray(sigma_hx, dtype=float)
    sigma_hz = np.asarray(sigma_hz, dtype=float)
    cal_arr = None if C is None else np.asarray(C, dtype=complex)
    if freq_mask is not None:
        m = np.asarray(freq_mask, dtype=bool)
        obs_hx, obs_hz = obs_hx[m], obs_hz[m]
        sigma_hx, sigma_hz = sigma_hx[m], sigma_hz[m]
        if cal_arr is not None:
            cal_arr = cal_arr[m]
    cal = cal_arr[:, None] if cal_arr is not None else 1.0

    res_x = (cal * hx_pred - obs_hx) / np.maximum(sigma_hx[:, None], 1e-300)
    res_z = (cal * hz_pred - obs_hz) / np.maximum(sigma_hz[:, None], 1e-300)

    m1, m2 = np.isfinite(res_x), np.isfinite(res_z)
    if not np.any(m1) and not np.any(m2):
        return 1e12

    mis = 0.0
    if np.any(m1):
        mis += float(w_hxh) * float(np.nansum(np.where(m1, np.abs(res_x) ** 2, np.nan)))
    if np.any(m2):
        mis += float(w_hxhz) * float(np.nansum(np.where(m2, np.abs(res_z) ** 2, np.nan)))

    if reg_lambda > 0.0 and n_layers > 1:
        lrho = np.asarray(params[:n_layers], dtype=float)
        mis += float(reg_lambda) * float(np.mean(np.diff(lrho) ** 2))

    return mis


def build_bounds(n_layers, log10_rho_min, log10_rho_max, log10_thk_min, log10_thk_max):
    b = [(float(log10_rho_min), float(log10_rho_max)) for _ in range(int(n_layers))]
    for _ in range(int(n_layers) - 1):
        b.append((float(log10_thk_min), float(log10_thk_max)))
    return b


__all__ = [
    "build_bounds",
    "complex_gain_objective",
    "forward_analytic_for_tx",
    "unpack_model_params",
]
