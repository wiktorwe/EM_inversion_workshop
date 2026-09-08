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

from pathlib import Path
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


# ---------------------------------------------------------------------------
# The 2x2 magnetic tensor
# ---------------------------------------------------------------------------
#
# Each component is one (source, receiver) pair. Both receiver components are
# recorded by every FDTD run, so the two SOURCE runs (Kx and Kz) between them
# give all four:
#
#     Cxx = Hx from Kx      Cxz = Hz from Kx
#     Czx = Hx from Kz      Czz = Hz from Kz
#
# The inversion used to fit only Cxx and Cxz - Hx and Hz from a single Kx source
# - because `forward_1d_gains` was hardcoded to the Kx solver. Fitting the full
# tensor needs one forward call per SOURCE (not per component), so the cost is
# 2x, not 4x.
TENSOR_COMPONENTS = {
    "Cxx": ("HX", "HX"),
    "Cxz": ("HX", "HZ"),
    "Czx": ("HZ", "HX"),
    "Czz": ("HZ", "HZ"),
}
DEFAULT_COMPONENTS = ("Cxx", "Cxz")          # the historical Kx-only pair


def forward_tensor_for_tx(params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
                          components=DEFAULT_COMPONENTS, n_nodes=120, freq_mask=None,
                          snap_dz=None):
    """Analytic gains for the requested tensor components.

    Returns ``{component: complex [nfreq_sel, nrx]}``. One solver call per
    distinct SOURCE in `components`, so asking for all four costs two forwards.
    """
    rho, thk, _ = unpack_model_params(params, n_layers, z_start_rel, z_end_rel,
                                      snap_dz=snap_dz)
    tx_z = float(tx_entry["tx_z"])
    off_x = np.asarray(tx_entry["off_x"], dtype=float)
    rx_depth_m = tx_z + np.asarray(tx_entry["off_z"], dtype=float)
    freqs = np.asarray(tx_entry["freqs"], dtype=float)
    if freq_mask is not None:
        freqs = freqs[np.asarray(freq_mask, dtype=bool)]

    unknown = [c for c in components if c not in TENSOR_COMPONENTS]
    if unknown:
        raise ValueError(f"Unknown tensor component(s) {unknown}; "
                         f"expected from {sorted(TENSOR_COMPONENTS)}")

    out = {}
    for src in sorted({TENSOR_COMPONENTS[c][0] for c in components}):
        hx, hz = forward_1d_gains(rho, thk, freqs, off_x, tx_z, rx_depth_m, eps_r,
                                  n_nodes=n_nodes, source_field=src)
        for c in components:
            s_c, r_c = TENSOR_COMPONENTS[c]
            if s_c == src:
                out[c] = hx if r_c == "HX" else hz
    return out


def tensor_objective(params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
                     reg_lambda, cal, components=DEFAULT_COMPONENTS, weights=None,
                     freq_mask=None, snap_dz=None):
    """Complex-gain misfit summed over the requested tensor components.

    `cal` carries the per-SOURCE calibration and the per-COMPONENT uncertainty:

        cal["C"][source]      complex [nfreq]  - FDTD/analytic scale for that source
        cal["sigma"][comp]    float   [nfreq]  - noise on that component

    Both are per-source for a reason: C is fitted separately for Kx and Kz, and
    the residual scatter differs sharply between components because the CROSS
    terms are near-nulls (measured 0.06 % on Hx from Kx against 2.6 % on Hz from
    Kx). Using one band- or component-averaged sigma would let the well-resolved
    co-components be swamped by the noisy cross terms, or vice versa.
    """
    try:
        pred = forward_tensor_for_tx(params, tx_entry, n_layers, z_start_rel, z_end_rel,
                                     eps_r, components=components, freq_mask=freq_mask,
                                     snap_dz=snap_dz)
    except ForwardRejected:
        return 1e12

    m = None if freq_mask is None else np.asarray(freq_mask, dtype=bool)
    weights = weights or {}
    mis = 0.0
    any_finite = False
    for comp in components:
        obs = tx_entry["obs"].get(comp)
        if obs is None:
            continue
        obs = np.asarray(obs, dtype=complex)
        sig = np.asarray(cal["sigma"][comp], dtype=float)
        src = TENSOR_COMPONENTS[comp][0]
        C = cal["C"].get(src)
        C = None if C is None else np.asarray(C, dtype=complex)
        if m is not None:
            obs, sig = obs[m], sig[m]
            if C is not None:
                C = C[m]
        scale = C[:, None] if C is not None else 1.0
        res = (scale * pred[comp] - obs) / np.maximum(sig[:, None], 1e-300)
        good = np.isfinite(res)
        if not np.any(good):
            continue
        any_finite = True
        mis += float(weights.get(comp, 1.0)) * float(
            np.nansum(np.where(good, np.abs(res) ** 2, np.nan))
        )
    if not any_finite:
        return 1e12

    if reg_lambda > 0.0 and n_layers > 1:
        lrho = np.asarray(params[:n_layers], dtype=float)
        mis += float(reg_lambda) * float(np.mean(np.diff(lrho) ** 2))
    return mis


def n_tensor_data(tx_entry, components=DEFAULT_COMPONENTS, freq_mask=None):
    """Real-valued datum count, for turning a misfit into a reduced chi-squared."""
    nrx = np.asarray(tx_entry["off_x"]).size
    nfreq = np.asarray(tx_entry["freqs"]).size
    if freq_mask is not None:
        nfreq = int(np.count_nonzero(np.asarray(freq_mask, dtype=bool)))
    present = [c for c in components if tx_entry.get("obs", {}).get(c) is not None]
    return 2 * nfreq * nrx * len(present)      # complex -> 2 real numbers


def tensor_calibration(meta_paths):
    """Per-source C and per-component sigma, assembled from setup metadata.

    `meta_paths` maps a source component ("HX"/"HZ") to the
    `setup_metadata.json` that holds its calibration. They may be the SAME file:
    `save_calibration_to_metadata` stores every calibration it has ever run under
    `fdtd_analytic_calibration_by_source`, keyed by source, so one Kx dataset's
    metadata typically carries both.

    sigma is resolved per COMPONENT, not per source or per band, because the
    residual scatter differs sharply between them - the cross terms are
    near-nulls (0.06 % on Hx-from-Kx against 2.6 % on Hz-from-Kx on this survey).
    """
    import json

    C, sigma, freqs, methods = {}, {}, None, {}
    for src, path in meta_paths.items():
        src = str(src).upper()
        meta = json.loads(Path(path).read_text())
        block = (meta.get("fdtd_analytic_calibration_by_source") or {}).get(src)
        if block is None:
            active = meta.get("fdtd_analytic_calibration") or {}
            if str(active.get("source_field", "HX")).upper() == src:
                block = active
        if block is None:
            raise KeyError(
                f"No calibration for source {src} in {path}. Run the Step 02 "
                f"calibration with 'cal source' set to {src}."
            )
        methods[src] = (str(block.get("method", "?")), float(block.get("rho_ohm_m", float("nan"))))
        C[src] = (np.asarray(block["C_hxhz_shared_real"], dtype=float)
                  + 1j * np.asarray(block["C_hxhz_shared_imag"], dtype=float))
        f = np.asarray(block["freqs_hz"], dtype=float)
        if freqs is None:
            freqs = f
        elif not np.allclose(freqs, f):
            raise ValueError(
                f"Calibrations disagree on frequencies: {freqs} vs {f}. Every "
                "tensor component must be calibrated on the same band."
            )
        for comp, (s_c, r_c) in TENSOR_COMPONENTS.items():
            if s_c != src:
                continue
            sigma[comp] = np.asarray(
                block["sigma_hx" if r_c == "HX" else "sigma_hz"], dtype=float
            )
    # Every component of one tensor must be calibrated on the SAME Earth model.
    # `save_calibration_to_metadata` keeps the last calibration per source, so
    # running (say) the lateral-average method for Kx and the homogeneous one for
    # Kz leaves a metadata file that looks complete but mixes a 28 Ohm-m
    # reference with a 1 Ohm-m one - different C AND different sigma scales,
    # silently. Measured when this first happened: the Czz residual came out at
    # 8 sigma while every other component sat under 0.2, purely because its sigma
    # came from the wrong Earth.
    distinct = set(methods.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{s}: {m} (rho_ref={r:.4g})" for s, (m, r) in sorted(methods.items()))
        raise ValueError(
            "Tensor components are calibrated on DIFFERENT Earth models - "
            f"{detail}. Re-run the Step 02 calibration with the same method for "
            "every source component before inverting the tensor."
        )
    return {"C": C, "sigma": sigma, "freqs_hz": freqs, "method": next(iter(distinct))[0]}


def load_tensor_features(dataset_dirs, freqs_hz=None, n_periods_extract=None):
    """Per-Tx observations for every available tensor component.

    `dataset_dirs` maps a source component to the forward run that used it. Both
    receiver components come out of each run, so two directories give all four
    tensor components. Geometry is cross-checked between sources: a silent
    mismatch in transmitter positions or offsets would pair the wrong traces and
    show up only as an unexplained misfit.
    """
    import json
    from scripts.modules.fd_visualization import compute_gains_for_fd_outputs

    per_src, ref = {}, None
    for src, d in dataset_dirs.items():
        src, d = str(src).upper(), Path(d)
        meta = json.loads((d / "setup_metadata.json").read_text())
        f = np.asarray(freqs_hz if freqs_hz is not None else meta["flist_hz"], dtype=float)
        npx = float(n_periods_extract if n_periods_extract is not None
                    else meta["n_periods_extract"])
        g = compute_gains_for_fd_outputs(
            d / "Data" / "Hxshot.rss", d / "Data" / "Hzshot.rss", d / "wav2d.rss",
            freqs=f, f_min_hz=float(meta["f_min_hz"]), n_periods_extract=npx,
        )
        per_src[src] = {"g": g, "meta": meta, "freqs": f}
        geo = g["geometry"]
        key = (np.round(np.asarray(geo["tx_unique"], float), 4).tolist(),
               np.round(np.asarray(geo["rx_x"], float), 4).tolist())
        if ref is None:
            ref = (src, key)
        elif key != ref[1]:
            raise ValueError(
                f"Dataset {src} has a different survey geometry from {ref[0]}. "
                "Every tensor component must come from the same survey."
            )

    first = next(iter(per_src.values()))
    geo = first["g"]["geometry"]
    tx_idx = np.asarray(geo["tx_idx_per_trace"], dtype=int)
    tx_unique = np.asarray(geo["tx_unique"], dtype=float)
    rx_x = np.asarray(geo["rx_x"], dtype=float)
    rx_z = np.asarray(geo["rx_z"], dtype=float)

    tx_data = {}
    for t in np.unique(tx_idx):
        tr = np.where(tx_idx == int(t))[0]
        obs = {}
        for src, blk in per_src.items():
            for comp, (s_c, r_c) in TENSOR_COMPONENTS.items():
                if s_c != src:
                    continue
                obs[comp] = np.asarray(
                    blk["g"]["Hx" if r_c == "HX" else "Hz"]["gain"][:, tr], dtype=complex
                )
        tx_data[int(t)] = {
            "tx_id": int(t),
            "tx_x": float(tx_unique[int(t), 0]),
            "tx_z": float(tx_unique[int(t), 1]),
            "off_x": rx_x[tr] - float(tx_unique[int(t), 0]),
            "off_z": rx_z[tr] - float(tx_unique[int(t), 1]),
            "freqs": first["freqs"],
            "obs": obs,
            # kept so the historical single-source objective still works
            "obs_hx_gain": obs.get("Cxx"),
            "obs_hz_gain": obs.get("Cxz"),
        }
    return {"tx_data": tx_data, "components": sorted(obs), "freqs": first["freqs"],
            "sources": sorted(per_src)}


def build_bounds(n_layers, log10_rho_min, log10_rho_max, log10_thk_min, log10_thk_max):
    b = [(float(log10_rho_min), float(log10_rho_max)) for _ in range(int(n_layers))]
    for _ in range(int(n_layers) - 1):
        b.append((float(log10_thk_min), float(log10_thk_max)))
    return b


__all__ = [
    "DEFAULT_COMPONENTS",
    "TENSOR_COMPONENTS",
    "build_bounds",
    "forward_tensor_for_tx",
    "load_tensor_features",
    "tensor_calibration",
    "n_tensor_data",
    "tensor_objective",
    "complex_gain_objective",
    "forward_analytic_for_tx",
    "unpack_model_params",
]
