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


def tensor_objective_parts(params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
                           reg_lambda, cal, components=DEFAULT_COMPONENTS, weights=None,
                           freq_mask=None, snap_dz=None):
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
                                     snap_dz=snap_dz)
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
        return _REJECT_COST, 0.0, _REJECT_COST

    reg_norm = 0.0
    if n_layers > 1:
        lrho = np.asarray(params[:n_layers], dtype=float)
        reg_norm = float(np.mean(np.diff(lrho) ** 2))
    total = mis + (float(reg_lambda) * reg_norm if reg_lambda > 0.0 else 0.0)
    return mis, reg_norm, total


def tensor_objective(params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
                     reg_lambda, cal, components=DEFAULT_COMPONENTS, weights=None,
                     freq_mask=None, snap_dz=None):
    """Scalar misfit for the optimisers - `tensor_objective_parts`'s total."""
    return tensor_objective_parts(
        params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r, reg_lambda, cal,
        components=components, weights=weights, freq_mask=freq_mask, snap_dz=snap_dz,
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
            f"calibration with 'cal source' set to {src}."
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


def resolve_tensor_calibration(cfg, setup_meta_path):
    """Per-source C and per-component sigma for the components `cfg` selects.

    A Kx-only selection reuses the flat active calibration
    (`fdtd_analytic_calibration.calibration_for_inversion`) exactly as before, so
    the historical two-component path is bit-for-bit unchanged. Anything
    involving Kz is assembled from `fdtd_analytic_calibration_by_source`, where
    Step 02 stores every calibration it has run.

    This lived in notebook 05 and so was unreachable from `inversion_tuning`,
    which is why the tuners kept their own Kx-only calibration handling and
    tuned a different objective from the one the run minimised.
    """
    comps = tuple(cfg.get("components") or DEFAULT_COMPONENTS)
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
    return {"tx_data": tx_data, "components": sorted(obs), "freqs": freqs,
            "sources": sources, "geometry": geo_keep}


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
    "unpack_model_params",
]
