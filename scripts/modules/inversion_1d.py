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
from typing import Mapping, Optional, Sequence

import numpy as np

from scripts.modules.analytic_1d_forward import ForwardRejected, forward_1d_gains

# Finite sentinel for a model the analytic solver refuses. It must be finite:
# a NaN or inf would make `differential_evolution` and the L-curve geometry
# undefined rather than merely bad.
_REJECT_COST = 1e12


def unpack_model_params(params, n_layers, z_start_rel, z_end_rel):
    """(rho, thickness, interface depths) from the optimiser's parameter vector.

    Resistivities are optimised in log10 space; thicknesses too, then rescaled
    so they exactly fill the `[z_start_rel, z_end_rel]` window - so the depth
    window is a hard constraint rather than something the optimiser can drift
    out of, and the total span is a CONSTANT of the run rather than a property
    of the candidate.

    This used to take `snap_dz`/`snap_origin_rel` and snap the interface depths
    onto one grid, in the tx-RELATIVE frame. Snapping now lives in
    `analytic_1d_forward.snap_interfaces_to_grid` and is applied inside
    `forward_1d_gains`, in ABSOLUTE depth and PER FREQUENCY, because each
    dataset of an acquisition matrix resamples the same `sg.rss` on its own `dx`
    and so quantises the same true interface to a different depth (measured
    6020.1 / 6020.875 / 6020.8 m at 2/4/6 kHz for a true interface at 6020.5 m).
    One relative grid could only ever be right for one tone of a joint fit.
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
    else:
        thk = np.array([], dtype=float)
        depth = np.array([], dtype=float)
    return rho, thk, depth


def blocky_layers_from_trace(rho_cells, z0, dz, rtol=1e-5, atol=0.0):
    """Merge adjacent equal-rho runs into `(interface_depths_abs, res)`.

    `depth` holds the interior interfaces only (empymod's convention), absolute
    and positive down; `res` is one resistivity per layer.

    An interface sits at the MIDPOINT between the last sample of one run and the
    first of the next: `z0 + (k + 1/2)*dz`. `.rss` samples are NODES at
    `o + k*d`, not cell tops - so treating the interface as the "cell bottom"
    `z0 + (k+1)*dz` puts every one of them HALF A CELL TOO DEEP. That is what
    the version of this in notebook 05 did, and it was wrong by exactly dz/2 in
    every dataset of this survey (measured 0.4 / 0.475 / 0.7 m at 6/4/2 kHz
    against `scripts/experiments/interface_snapping.py`, which reads the step
    midpoint out of `sg.rss` independently). Half a cell moves |Hz| by 4.9-6.6 %
    here, against a 3 % uncertainty floor - so this was a real bias in every
    true-model overlay, not a cosmetic offset.
    """
    rho = np.asarray(rho_cells, dtype=float).ravel()
    nz = rho.size
    if nz == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    boundary = float(z0) + (np.arange(nz, dtype=float) + 0.5) * float(dz)
    is_new = np.ones(nz, dtype=bool)
    is_new[1:] = ~np.isclose(rho[1:], rho[:-1], rtol=rtol, atol=atol)
    run_starts = np.flatnonzero(is_new)

    res = np.empty(run_starts.size, dtype=float)
    depth = np.empty(max(0, run_starts.size - 1), dtype=float)
    for j, s in enumerate(run_starts):
        e = nz - 1 if j == run_starts.size - 1 else int(run_starts[j + 1]) - 1
        res[j] = rho[int(s)]
        if j < run_starts.size - 1:
            depth[j] = boundary[int(e)]
    return depth, res


def true_model_layers_from_sg(sg_path, tx_x, tx_z, z_positive_up=False):
    """`(depth_rel, res)` for the TRUE model beneath a transmitter, from sg.rss.

    THE one implementation, so that an experiment and the notebook cannot drift
    apart the way three copies of the 1D misfit once did. Notebook 05 imports
    this rather than keeping its own reader. `depth_rel` is tx-relative, so it
    drops straight into the inversion's parameterisation.

    The column NEAREST `tx_x` is read, never the lateral average: averaging
    across the fault throw mixes two different interface depths into one.
    """
    # `read_sg_grid` is the one .rss conductivity reader - it already handles
    # the 2D/3D layout and the `o + k*d` sample convention.
    from scripts.modules.multiscale_2d import read_sg_grid

    g = read_sg_grid(sg_path)
    xg, zg = np.asarray(g["x"], float), np.asarray(g["z"], float)
    rho_grid = 1.0 / np.clip(np.asarray(g["sigma"], float), 1e-12, 1e12)
    ix = int(np.argmin(np.abs(xg - float(tx_x)))) if xg.size else 0
    rho_trace = np.asarray(rho_grid[ix, :], dtype=float)

    z0 = float(g["oz"])
    dz = float(g["dz"])
    if z_positive_up:
        z0, dz = -z0, -dz
    if dz <= 0.0:
        raise ValueError("True-model vertical dz must be > 0 after mapping.")

    depth_abs, res = blocky_layers_from_trace(rho_trace, z0=z0, dz=dz)
    return np.asarray(depth_abs, dtype=float) - float(tx_z), res


def _slice_per_freq(value, freqs_full, freq_mask):
    """A per-frequency quantity sliced to the SELECTED frequencies.

    `value` may be None, a scalar, or one value per frequency of
    `tx_entry["freqs"]`. A frequency subset (one stage of the multi-scale
    ladder) must slice it with the same mask, or the analytic forward is
    evaluated with another frequency's number.

    Used for `eps_r`, `snap_dz` and `snap_origin_m` - every quantity Step 01
    sizes PER DATASET. It was written for `eps_r` alone and named for it; the
    rule is the same for all of them, so it is named for the rule now.
    """
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1 or freq_mask is None:
        return float(arr[0]) if arr.size == 1 else arr
    m = np.asarray(freq_mask, dtype=bool)
    if arr.size != m.size:
        raise ValueError(
            f"a per-frequency value has {arr.size} entries but tx_entry "
            f"has {m.size} frequencies"
        )
    return arr[m]


def forward_analytic_for_tx(params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
                            n_nodes=120, freq_mask=None, snap_dz=None,
                            snap_origin_m=None):
    """Analytic (Hx, Hz) complex channel gain for a candidate model, via
    `analytic_1d_forward.forward_1d_gains` - the validated 2D magnetic
    line-source solver, exact counterpart of the FDTD survey (see that
    module's docstring). Raises `ForwardRejected` on an unevaluable model -
    callers must catch it (never let it silently become NaN).

    `freq_mask` restricts the forward to a subset of `tx_entry["freqs"]`. That
    is what makes one stage of the multi-scale ladder a SLICE rather than a
    rewrite: nothing else about the misfit changes between stages.

    `snap_dz`/`snap_origin_m` are per-frequency like `eps_r`, and are sliced
    with the same mask - a stage that fits 2 kHz must snap onto the 2 kHz grid.
    """
    rho, thk, _ = unpack_model_params(params, n_layers, z_start_rel, z_end_rel)
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
    freqs_full = np.asarray(tx_entry["freqs"], dtype=float)
    freqs = freqs_full
    if freq_mask is not None:
        freqs = freqs_full[np.asarray(freq_mask, dtype=bool)]
    eps = _slice_per_freq(eps_r, freqs_full, freq_mask)
    return forward_1d_gains(
        rho, thk, freqs, off_x, tx_z, rx_depth_m, eps, n_nodes=n_nodes,
        snap_dz=_slice_per_freq(snap_dz, freqs_full, freq_mask),
        snap_origin_m=_slice_per_freq(snap_origin_m, freqs_full, freq_mask),
    )


def complex_gain_objective(
    params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
    reg_lambda, w_hxh, w_hxhz, sigma_hx, sigma_hz, C=None, freq_mask=None, snap_dz=None,
    snap_origin_m=None,
):
    """Complex-gain misfit: ((C*pred - obs)/sigma) summed in quadrature over
    frequency and receiver, for both Hx and Hz, plus a Tikhonov first-
    difference penalty on log10(rho).

    `C` is the FDTD-to-analytic scale, shared by all Tx and both Hx/Hz. It is
    COMPUTED (`fd_error_model.analytic_C`), not fitted from a calibration run;
    notebook 02's run validates it rather than producing it.

    `freq_mask` (boolean, length nfreq) restricts the misfit to a subset of the
    frequencies - one stage of a multi-scale ladder. `sigma_hx`, `sigma_hz` and
    `C` are per-frequency arrays and are sliced with the same mask, so each
    stage automatically gets that frequency's OWN calibration scatter rather
    than a band-averaged one.
    """
    try:
        hx_pred, hz_pred = forward_analytic_for_tx(
            params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
            freq_mask=freq_mask, snap_dz=snap_dz, snap_origin_m=snap_origin_m)
    except ForwardRejected:
        return _REJECT_COST

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
        return _REJECT_COST

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
#
# The map itself lives in `fd_visualization`, because the plot GUIs need it too
# (to label the freq x source x receiver view selector) and this module already
# imports that one - defining it in both places is how the two would drift.
from scripts.modules.fd_visualization import TENSOR_COMPONENTS  # noqa: E402
DEFAULT_COMPONENTS = ("Cxx", "Cxz")          # the historical Kx-only pair


def forward_tensor_for_tx(params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
                          components=DEFAULT_COMPONENTS, n_nodes=120, freq_mask=None,
                          snap_dz=None, snap_origin_m=None):
    """Analytic gains for the requested tensor components.

    Returns ``{component: complex [nfreq_sel, nrx]}``. One solver call per
    distinct SOURCE in `components`, so asking for all four costs two forwards.
    """
    rho, thk, _ = unpack_model_params(params, n_layers, z_start_rel, z_end_rel)
    tx_z = float(tx_entry["tx_z"])
    off_x = np.asarray(tx_entry["off_x"], dtype=float)
    rx_depth_m = tx_z + np.asarray(tx_entry["off_z"], dtype=float)
    freqs_full = np.asarray(tx_entry["freqs"], dtype=float)
    freqs = freqs_full
    if freq_mask is not None:
        freqs = freqs_full[np.asarray(freq_mask, dtype=bool)]
    eps = _slice_per_freq(eps_r, freqs_full, freq_mask)

    unknown = [c for c in components if c not in TENSOR_COMPONENTS]
    if unknown:
        raise ValueError(f"Unknown tensor component(s) {unknown}; "
                         f"expected from {sorted(TENSOR_COMPONENTS)}")

    out = {}
    for src in sorted({TENSOR_COMPONENTS[c][0] for c in components}):
        hx, hz = forward_1d_gains(
            rho, thk, freqs, off_x, tx_z, rx_depth_m, eps, n_nodes=n_nodes,
            source_field=src,
            snap_dz=_slice_per_freq(snap_dz, freqs_full, freq_mask),
            snap_origin_m=_slice_per_freq(snap_origin_m, freqs_full, freq_mask),
        )
        for c in components:
            s_c, r_c = TENSOR_COMPONENTS[c]
            if s_c == src:
                out[c] = hx if r_c == "HX" else hz
    return out


def tensor_objective_parts(params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
                           reg_lambda, cal, components=DEFAULT_COMPONENTS, weights=None,
                           freq_mask=None, snap_dz=None, snap_origin_m=None):
    """`(data_misfit, reg_norm, total)` for the requested tensor components.

    THE one implementation. `tensor_objective` returns only `total` from it, and
    `inversion_tuning.split_objective` returns all three - so the DE-budget and
    lambda tuners decompose exactly the functional the run minimises. They used
    to carry their own copy that was hardwired to the Kx pair, which meant that
    with tensor components selected the tuners recommended a budget and a lambda
    for a DIFFERENT objective than the one being optimised.

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
                                     snap_dz=snap_dz, snap_origin_m=snap_origin_m)
    except ForwardRejected:
        return _REJECT_COST, 0.0, _REJECT_COST

    m = None if freq_mask is None else np.asarray(freq_mask, dtype=bool)
    weights = weights or {}
    mis = 0.0
    any_finite = False
    for comp in components:
        obs = tx_entry["obs"].get(comp)
        if obs is None:
            continue
        obs = np.asarray(obs, dtype=complex)
        # sigma may be PER FREQUENCY [nfreq] (the fitted calibration) or PER
        # FREQUENCY AND RECEIVER [nfreq, nrx] (the analytic budget). A relative
        # budget is scaled by THIS transmitter's own data here, rather than
        # being baked in per Tx, so one calibration object serves every Tx.
        rel = (cal.get("sigma_rel") or {}).get(comp)
        if rel is not None:
            rel = np.asarray(rel, dtype=float)
            sig = rel * np.abs(obs)
            med = np.nanmedian(np.where(np.abs(obs) > 0, np.abs(obs), np.nan), axis=1)
            med = np.where(np.isfinite(med), med, 1.0)[:, None]
            sig = np.maximum(sig, rel * med * 1e-2)
        else:
            sig = np.asarray(cal["sigma"][comp], dtype=float)
        src = TENSOR_COMPONENTS[comp][0]
        C = cal["C"].get(src)
        C = None if C is None else np.asarray(C, dtype=complex)
        if m is not None:
            obs = obs[m]
            sig = sig[m]
            if C is not None:
                C = C[m]
        # `sig[:, None]` on an ALREADY 2-D sigma would silently insert an axis
        # and broadcast the receivers against each other rather than raising -
        # so decide by shape, explicitly. A per-frequency sigma legitimately
        # becomes (nfreq, 1) and broadcasts across receivers; a per-receiver one
        # must already match.
        if sig.ndim == 1:
            sig = sig[:, None]
        if sig.shape not in (obs.shape, (obs.shape[0], 1)):
            raise ValueError(
                f"sigma for {comp} has shape {sig.shape}, expected {obs.shape} "
                f"(per receiver) or ({obs.shape[0]}, 1) (per frequency)")
        scale = C[:, None] if C is not None else 1.0
        res = (scale * pred[comp] - obs) / np.maximum(sig, 1e-300)
        good = np.isfinite(res)
        if not np.any(good):
            continue
        any_finite = True
        mis += float(weights.get(comp, 1.0)) * float(
            np.nansum(np.where(good, np.abs(res) ** 2, np.nan))
        )
    if not any_finite:
        return _REJECT_COST, 0.0, _REJECT_COST

    reg_norm = 0.0
    if n_layers > 1:
        lrho = np.asarray(params[:n_layers], dtype=float)
        reg_norm = float(np.mean(np.diff(lrho) ** 2))
    total = mis + (float(reg_lambda) * reg_norm if reg_lambda > 0.0 else 0.0)
    return mis, reg_norm, total


def tensor_objective(params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
                     reg_lambda, cal, components=DEFAULT_COMPONENTS, weights=None,
                     freq_mask=None, snap_dz=None, snap_origin_m=None):
    """Scalar misfit for the optimisers - `tensor_objective_parts`'s total."""
    return tensor_objective_parts(
        params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r, reg_lambda, cal,
        components=components, weights=weights, freq_mask=freq_mask, snap_dz=snap_dz,
        snap_origin_m=snap_origin_m,
    )[2]


def n_tensor_data(tx_entry, components=DEFAULT_COMPONENTS, freq_mask=None):
    """Real-valued datum count, for turning a misfit into a reduced chi-squared."""
    nrx = np.asarray(tx_entry["off_x"]).size
    nfreq = np.asarray(tx_entry["freqs"]).size
    if freq_mask is not None:
        nfreq = int(np.count_nonzero(np.asarray(freq_mask, dtype=bool)))
    present = [c for c in components if tx_entry.get("obs", {}).get(c) is not None]
    return 2 * nfreq * nrx * len(present)      # complex -> 2 real numbers


def _calibration_blocks(path, src):
    """Every stored calibration block for one source in one metadata file."""
    import json

    meta = json.loads(Path(path).read_text())
    block = (meta.get("fdtd_analytic_calibration_by_source") or {}).get(src)
    if block is None:
        active = meta.get("fdtd_analytic_calibration") or {}
        if str(active.get("source_field", "HX")).upper() == src:
            block = active
    if block is None:
        raise KeyError(
            f"No calibration for source {src} in {path}. Run the Step 02 "
            f"calibration batch - it calibrates EVERY dataset with the source "
            f"it was modelled with, so a {src} block appears once Step 01 has "
            f"built a {src} dataset. There is no per-source calibration control "
            f"to set (RULE 2)."
        )
    return block


def tensor_calibration(meta_paths):
    """Per-source C and per-component sigma, assembled from setup metadata.

    `meta_paths` maps a source component ("HX"/"HZ") to the
    `setup_metadata.json` that holds its calibration - or to a SEQUENCE of them,
    one per single-frequency dataset. They may be the SAME file:
    `save_calibration_to_metadata` stores every calibration it has ever run under
    `fdtd_analytic_calibration_by_source`, keyed by source, so one Kx dataset's
    metadata typically carries both.

    Per-frequency datasets each carry their OWN calibration, fitted on their own
    grid, and they genuinely differ - measured +2.04 % at 1 kHz against the
    broadband value, against 0.39 % scatter, with a same-grid control
    reproducing to 0.001 %. So they are assembled frequency by frequency rather
    than averaged, and passing a list of per-frequency metadata files is the
    supported way to invert a per-frequency acquisition matrix.

    sigma is resolved per COMPONENT, not per source or per band, because the
    residual scatter differs sharply between them - the cross terms are
    near-nulls (0.06 % on Hx-from-Kx against 2.6 % on Hz-from-Kx on this survey).
    """
    # freq -> source -> block, so both consistency checks below are per-frequency
    by_freq = {}
    for src, paths in meta_paths.items():
        src = str(src).upper()
        if isinstance(paths, (str, Path)):
            paths = [paths]
        for path in paths:
            block = _calibration_blocks(path, src)
            freqs_b = np.asarray(block["freqs_hz"], dtype=float)
            C_b = (np.asarray(block["C_hxhz_shared_real"], dtype=float)
                   + 1j * np.asarray(block["C_hxhz_shared_imag"], dtype=float))
            for i, f in enumerate(freqs_b):
                slot = by_freq.setdefault(float(f), {})
                if src in slot:
                    raise ValueError(
                        f"Source {src} has two calibrations for {f:g} Hz "
                        f"(the second from {path}). Pass one metadata file per "
                        f"frequency per source."
                    )
                slot[src] = {
                    "C": complex(C_b[i]),
                    "sigma_hx": float(np.asarray(block["sigma_hx"], dtype=float)[i]),
                    "sigma_hz": float(np.asarray(block["sigma_hz"], dtype=float)[i]),
                    "earth": (str(block.get("method", "?")),
                              float(block.get("rho_ohm_m", float("nan")))),
                    "path": str(path),
                }

    wanted = {str(s).upper() for s in meta_paths}
    freqs = np.asarray(sorted(by_freq), dtype=float)
    if freqs.size == 0:
        raise ValueError("No calibrations found in the given metadata.")

    for f in freqs:
        slot = by_freq[float(f)]
        missing = sorted(wanted - set(slot))
        if missing:
            raise ValueError(
                f"No calibration at {f:g} Hz for source(s) {missing}. Every tensor "
                "component must be calibrated on the same band."
            )
        # Every component of one tensor must be calibrated on the SAME Earth
        # model AT THAT FREQUENCY. `save_calibration_to_metadata` keeps the last
        # calibration per source, so running (say) the lateral-average method for
        # Kx and the homogeneous one for Kz leaves a metadata file that looks
        # complete but mixes a 28 Ohm-m reference with a 1 Ohm-m one - different
        # C AND different sigma scales, silently. Measured when this first
        # happened: the Czz residual came out at 8 sigma while every other
        # component sat under 0.2, purely because its sigma came from the wrong
        # Earth.
        #
        # The comparison is deliberately WITHIN a frequency, not across the band.
        # Per-frequency datasets legitimately differ in rho_ref (measured
        # 26.09 / 27.60 / 29.04 / 29.85 Ohm-m at 1/2/4/6 kHz) because each
        # frequency's grid resamples the same sg.rss differently. Comparing those
        # across frequencies would reject a correct setup.
        distinct = {slot[s]["earth"] for s in wanted}
        if len(distinct) > 1:
            detail = ", ".join(
                f"{s}: {slot[s]['earth'][0]} (rho_ref={slot[s]['earth'][1]:.4g})"
                for s in sorted(wanted)
            )
            raise ValueError(
                f"At {f:g} Hz the tensor components are calibrated on DIFFERENT "
                f"Earth models - {detail}. Re-run the Step 02 calibration with the "
                "same method for every source component before inverting the tensor."
            )

    C = {s: np.asarray([by_freq[float(f)][s]["C"] for f in freqs], dtype=complex)
         for s in wanted}
    sigma = {}
    for comp, (s_c, r_c) in TENSOR_COMPONENTS.items():
        if s_c not in wanted:
            continue
        key = "sigma_hx" if r_c == "HX" else "sigma_hz"
        sigma[comp] = np.asarray([by_freq[float(f)][s_c][key] for f in freqs], dtype=float)

    method = by_freq[float(freqs[0])][sorted(wanted)[0]]["earth"][0]
    return {"C": C, "sigma": sigma, "freqs_hz": freqs, "method": method}


def component_weights(cfg):
    """Per-component weights from the GUI's two knobs.

    `w_hxh` and `w_hxhz` weight the Hx- and Hz-RECEIVER components, whichever
    source they came from, so the two historical knobs keep their meaning as the
    tensor grows: `w_hxh` -> Cxx and Czx, `w_hxhz` -> Cxz and Czz.
    """
    keys = {"HX": "w_hxh", "HZ": "w_hxhz"}
    return {c: float(cfg.get(keys[recv], 1.0)) for c, (_src, recv) in TENSOR_COMPONENTS.items()}


def analytic_tensor_calibration(cfg, tx_entry, components=DEFAULT_COMPONENTS):
    """Per-source C and a per-component RELATIVE error budget, computed.

    No FDTD run, no fitting. `C = dx*dz*s(order)` and the budget is assembled
    from `fd_error_model` - see that module for what is in it and why.

    Returns the same shape as `tensor_calibration` plus `sigma_rel`:

        C          {source: complex [nfreq]}   - real-valued, but complex dtype
                                                 so every consumer is unchanged
        sigma_rel  {component: [nfreq, nrx]}   - RELATIVE, scaled by each Tx's
                                                 own |obs| in the objective
        sigma      {component: [nfreq, nrx]}   - absolute, for THIS tx_entry

    Why relative: the fitted sigma was one absolute number per frequency,
    inherited from the calibration run's amplitude scale and constant across
    offsets - so the near offset dominated the misfit and the far offset was
    effectively ignored. A relative budget weights them as the physics does.
    """
    from scripts.modules import fd_error_model as fem

    comps = tuple(components)
    freqs = np.asarray(tx_entry["freqs"], dtype=float).reshape(-1)
    off_x = np.asarray(tx_entry["off_x"], dtype=float).reshape(-1)
    order = int(cfg["fd_order"])
    dx = np.asarray(cfg["dx"], dtype=float).reshape(-1)
    if dx.size == 1:
        dx = np.full(freqs.size, float(dx[0]))
    if dx.size != freqs.size:
        raise ValueError(f"dx has {dx.size} entries but there are {freqs.size} frequencies")

    C = {s_: np.asarray([fem.analytic_C(float(d), order) for d in dx], dtype=complex)
         for s_ in sorted({TENSOR_COMPONENTS[c][0] for c in comps})}

    # Snapping and the quantisation term are ALTERNATIVES. If the candidate is
    # snapped onto the grid the error is gone from the residual, so charging for
    # it in sigma too is double-counting - measured worse (0.9357 vs 0.8417).
    if cfg.get("snap_dz") is not None:
        quant = {"HX": np.zeros(freqs.size), "HZ": np.zeros(freqs.size)}
    else:
        quant = cfg.get("quantisation_rel")
    if quant is None:
        tx_z = float(tx_entry["tx_z"])
        quant = {"HX": np.empty(freqs.size), "HZ": np.empty(freqs.size)}
        eps = np.asarray(cfg["eps_r"], dtype=float).reshape(-1)
        if eps.size == 1:
            eps = np.full(freqs.size, float(eps[0]))
        for i in range(freqs.size):
            q = fem.interface_quantisation_rel(
                freqs[i:i + 1], off_x, tx_z, eps[i:i + 1], float(dx[i]),
                rx_depth_m=tx_z + np.asarray(tx_entry["off_z"], dtype=float))
            quant["HX"][i] = q["HX"][0]
            quant["HZ"][i] = q["HZ"][0]

    sigma_rel, sigma = {}, {}
    for c in comps:
        obs = tx_entry.get("obs", {}).get(c)
        if obs is None:
            continue
        recv = TENSOR_COMPONENTS[c][1]
        # dx is per frequency, so the geometric term is too - build it row by row
        # rather than handing `sigma_budget` a single dx it would apply to the
        # whole band.
        rows_rel, rows_sig = [], []
        for i in range(freqs.size):
            b = fem.sigma_budget(np.asarray(obs, dtype=complex)[i:i + 1], off_x,
                                 float(dx[i]), order,
                                 quantisation_rel=np.asarray(quant[recv])[i])
            rows_rel.append(b["rel"][0])
            rows_sig.append(b["sigma"][0])
        sigma_rel[c] = np.asarray(rows_rel, dtype=float)
        sigma[c] = np.asarray(rows_sig, dtype=float)

    return {"C": C, "sigma": sigma, "sigma_rel": sigma_rel,
            "freqs_hz": freqs, "method": "analytic",
            "notes": (f"C = dx*dz*s(order={order}), s={fem.stencil_consistency(order):.6f}; "
                      f"sigma from the analytic error budget (no FDTD calibration)")}


def resolve_tensor_calibration(cfg, setup_meta_path, tx_entry=None):
    """Per-source C and per-component sigma for the components `cfg` selects.

    DEFAULT: the ANALYTIC model (`analytic_tensor_calibration`) - no FDTD run,
    no fitting. `C = dx*dz*s(order)` is derived from the engine's source
    injection, and sigma is an explicit error budget. Requires `tx_entry` (the
    budget is relative to the data) and `cfg["dx"]` / `cfg["fd_order"]`.

    Set `cfg["calibration_source"] = "fitted"` to fall back to the FDTD-fitted
    values in `setup_metadata.json`. That escape hatch exists because the
    comparison between the two IS the validation - see
    `scripts/experiments/analytic_C_check.py` - not because the fitted path is
    an equal alternative: it is fitted on a geometry where Hz sits on a
    near-null, which is what made its sigma unusable for the cross-couplings.

    This lived in notebook 05 and so was unreachable from `inversion_tuning`,
    which is why the tuners kept their own Kx-only calibration handling and
    tuned a different objective from the one the run minimised.
    """
    comps = tuple(cfg.get("components") or DEFAULT_COMPONENTS)
    if str(cfg.get("calibration_source", "analytic")).lower() == "analytic":
        if tx_entry is None:
            raise ValueError(
                "The analytic calibration is relative to the data, so it needs a "
                "tx_entry. Pass one, or set cfg['calibration_source']='fitted'.")
        return analytic_tensor_calibration(cfg, tx_entry, components=comps)
    sources = sorted({TENSOR_COMPONENTS[c][0] for c in comps})
    cal = cfg.get("calibration")
    if sources == ["HX"] and cal is not None:
        return {"C": {"HX": cal.get("C")},
                "sigma": {"Cxx": np.asarray(cal["sigma_hx"], dtype=float),
                          "Cxz": np.asarray(cal["sigma_hz"], dtype=float)}}
    # `setup_meta_path` may be one path shared by every source (the historical
    # case - one dataset's metadata carries every calibration it has run), or a
    # PER-SOURCE mapping, which is what an acquisition matrix needs: each source
    # has its own datasets, and with per-frequency datasets each source has a
    # LIST of them. Handing one source's paths to another would silently
    # calibrate Kz against Kx's grid.
    if isinstance(setup_meta_path, Mapping):
        missing = [s for s in sources if s not in setup_meta_path]
        if missing:
            raise KeyError(
                f"No calibration metadata given for source(s) {missing}. "
                f"Have: {sorted(setup_meta_path)}."
            )
        return tensor_calibration({s: setup_meta_path[s] for s in sources})
    return tensor_calibration({s: setup_meta_path for s in sources})


def load_tensor_features(dataset_dirs, freqs_hz=None, n_periods_extract=None):
    """Per-Tx observations for every available tensor component.

    `dataset_dirs` maps a source component to the forward run that used it - or
    to a SEQUENCE of runs, one per single-frequency dataset. Both receiver
    components come out of each run, so two sources give all four tensor
    components, and a per-frequency matrix contributes its tones one dataset at
    a time.

    Passing a list is how a per-frequency acquisition matrix is inverted at
    once. Each dataset is extracted at ITS OWN frequency (from its own
    `flist_hz` and `n_periods_extract`) and the rows are then stacked in
    ascending frequency order, because a per-frequency dataset only contains
    the tone it was designed for - asking it for the whole band would read
    noise at three of the four.

    Geometry is cross-checked between every dataset: a silent mismatch in
    transmitter positions or offsets would pair the wrong traces and show up
    only as an unexplained misfit.
    """
    import json
    from scripts.modules.fd_visualization import compute_gains_for_fd_outputs

    want = None if freqs_hz is None else np.asarray(freqs_hz, dtype=float).reshape(-1)

    rows, geo_ref, ref_label = {}, None, None
    for src, dirs in dataset_dirs.items():
        src = str(src).upper()
        if isinstance(dirs, (str, Path)):
            dirs = [dirs]
        for d in dirs:
            d = Path(d)
            meta = json.loads((d / "setup_metadata.json").read_text())
            f_here = np.asarray(meta["flist_hz"], dtype=float).reshape(-1)
            if want is not None:
                f_here = np.asarray([f for f in f_here
                                     if np.any(np.isclose(want, f))], dtype=float)
                if f_here.size == 0:
                    continue
            npx = float(n_periods_extract if n_periods_extract is not None
                        else meta["n_periods_extract"])
            g = compute_gains_for_fd_outputs(
                d / "Data" / "Hxshot.rss", d / "Data" / "Hzshot.rss", d / "wav2d.rss",
                freqs=f_here, f_min_hz=float(meta["f_min_hz"]), n_periods_extract=npx,
            )
            geo = g["geometry"]
            key = (np.round(np.asarray(geo["tx_unique"], float), 4).tolist(),
                   np.round(np.asarray(geo["rx_x"], float), 4).tolist())
            if geo_ref is None:
                geo_ref, ref_label, geo_keep = key, f"{src}:{d.name}", geo
            elif key != geo_ref:
                raise ValueError(
                    f"Dataset {src}:{d.name} has a different survey geometry from "
                    f"{ref_label}. Every tensor component must come from the same survey."
                )
            for i, f in enumerate(f_here):
                slot = (src, float(f))
                if slot in rows:
                    raise ValueError(
                        f"Two datasets provide source {src} at {f:g} Hz (the second "
                        f"is {d}). Pass one dataset per frequency per source."
                    )
                rows[slot] = {"Hx": np.asarray(g["Hx"]["gain"][i, :], dtype=complex),
                              "Hz": np.asarray(g["Hz"]["gain"][i, :], dtype=complex)}

    if not rows:
        raise ValueError("No usable datasets in dataset_dirs.")

    sources = sorted({s for s, _ in rows})
    freqs = np.asarray(sorted({f for _, f in rows}), dtype=float)
    for s in sources:
        missing = [f for f in freqs if (s, float(f)) not in rows]
        if missing:
            raise ValueError(
                f"Source {s} has no dataset at {missing} Hz. Every component of one "
                "tensor must cover the same band."
            )

    tx_idx = np.asarray(geo_keep["tx_idx_per_trace"], dtype=int)
    tx_unique = np.asarray(geo_keep["tx_unique"], dtype=float)
    rx_x = np.asarray(geo_keep["rx_x"], dtype=float)
    rx_z = np.asarray(geo_keep["rx_z"], dtype=float)

    tx_data = {}
    for t in np.unique(tx_idx):
        tr = np.where(tx_idx == int(t))[0]
        obs = {}
        for comp, (s_c, r_c) in TENSOR_COMPONENTS.items():
            if s_c not in sources:
                continue
            obs[comp] = np.stack(
                [rows[(s_c, float(f))]["Hx" if r_c == "HX" else "Hz"][tr] for f in freqs]
            ).astype(complex)
        tx_data[int(t)] = {
            "tx_id": int(t),
            "tx_x": float(tx_unique[int(t), 0]),
            "tx_z": float(tx_unique[int(t), 1]),
            "rx_x": rx_x[tr],
            "rx_z": rx_z[tr],
            "off_x": rx_x[tr] - float(tx_unique[int(t), 0]),
            "off_z": rx_z[tr] - float(tx_unique[int(t), 1]),
            "freqs": freqs,
            "obs": obs,
            # kept so the historical single-source objective still works
            "obs_hx_gain": obs.get("Cxx"),
            "obs_hz_gain": obs.get("Cxz"),
        }
    # Per-TRACE gains too, stacked in frequency order, so a caller that wants the
    # `compute_gains_for_fd_outputs` shape (notebook 06's real-vs-synthetic
    # comparison) does not have to re-extract - and cannot re-extract the whole
    # band out of one per-frequency dataset, which is the bug this assembly
    # exists to prevent.
    gains = {}
    for s_ in sources:
        hx = np.stack([rows[(s_, float(f))]["Hx"] for f in freqs]).astype(complex)
        hz = np.stack([rows[(s_, float(f))]["Hz"] for f in freqs]).astype(complex)
        gains[s_] = {"Hx": hx, "Hz": hz}
    return {"tx_data": tx_data, "components": sorted(obs), "freqs": freqs,
            "sources": sources, "geometry": geo_keep, "gains": gains}


def build_bounds(n_layers, log10_rho_min, log10_rho_max, log10_thk_min, log10_thk_max):
    b = [(float(log10_rho_min), float(log10_rho_max)) for _ in range(int(n_layers))]
    for _ in range(int(n_layers) - 1):
        b.append((float(log10_thk_min), float(log10_thk_max)))
    return b


__all__ = [
    "DEFAULT_COMPONENTS",
    "TENSOR_COMPONENTS",
    "build_bounds",
    "resolve_tensor_calibration",
    "tensor_objective_parts",
    "forward_tensor_for_tx",
    "load_tensor_features",
    "tensor_calibration",
    "n_tensor_data",
    "tensor_objective",
    "complex_gain_objective",
    "forward_analytic_for_tx",
    "blocky_layers_from_trace",
    "true_model_layers_from_sg",
    "unpack_model_params",
]
