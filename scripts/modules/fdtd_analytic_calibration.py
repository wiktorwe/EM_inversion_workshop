"""Global FDTD-vs-analytic calibration for the workshop (homogeneous rho_min).

Computes a single Tx-independent complex scale C[f] per frequency such that
FDTD channel gain ≈ C * analytic channel gain, following rockem-suite's
convention (see validate_layered_1d_model/README.md: C = FDTD/analytic,
|C| ≈ dx² for homogeneous whole-space).

The calibration survey keeps production x-offsets but places receivers both
above and below the transmitter so Hx-source Hz is nonzero (rockem-suite
Hx validation uses rx_dz ≠ 0 for the same reason). Production rz0/tz0 is
unchanged.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scripts.modules.analytic_1d_forward import Layer1D
from scripts.modules.rockem_bridge import (
    config as rockem_config,
    magnetic_line_source_fields_layered,
    model as rockem_model,
    survey as rockem_survey,
)

VALIDATED_REL_ERROR_FLOOR = 0.03
CALIBRATION_SUBDIR = "calibration_homogeneous"
CALIBRATION_CFG_NAME = "mod_cal.cfg"
# Keep sources/receivers this many cells clear of the PML when the survey's own
# aperture margin is unavailable (rockem-suite gotchas: 8-16 cells).
MIN_PML_CLEARANCE_CELLS = 16
# rockem-suite Hx-source validation uses rx_dz = -20 m so Ey/Hz are nonzero in a
# homogeneous medium (they null exactly at the source depth). Calibration always
# places receivers both above and below the Tx by at least this amount (and at
# least two grid cells), independent of the production survey's rz0/tz0.
CALIBRATION_RX_DZ_M = 20.0
MIN_CALIBRATION_RX_DZ_CELLS = 2

_REQUIRED_META_FIELDS = (
    "rho_min_ohm_m",
    "eps_r_used",
    "dx_model_target_m",
    "dt_model_target_s",
    "tz0_m",
    "rz0_m",
    "rx0_m",
    "drx_m",
    "nrx",
)


def load_setup_metadata(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"setup metadata not found: {path}")
    return json.loads(path.read_text())


def _require_meta_fields(meta: Mapping[str, Any]) -> dict:
    missing = [k for k in _REQUIRED_META_FIELDS if k not in meta]
    if missing:
        raise KeyError(
            f"setup_metadata.json missing {missing}. Re-run 01_fw_setup 'Apply outputs' "
            "after updating the workshop to persist FD design bounds."
        )
    return dict(meta)


def calibration_run_dir(fwd_2d_dir: Path | str) -> Path:
    return Path(fwd_2d_dir) / CALIBRATION_SUBDIR


def calibration_rx_dz_m(meta: Mapping[str, Any]) -> float:
    """Vertical Tx–Rx separation used only for the homogeneous calibration survey."""
    dx = float(meta["dx_model_target_m"])
    return float(max(CALIBRATION_RX_DZ_M, MIN_CALIBRATION_RX_DZ_CELLS * dx))


def calibration_geometry(meta: Mapping[str, Any]) -> dict:
    """Purpose-sized homogeneous domain with the transmitter at its centre.

    Mirrors rockem-suite's own calibration reference
    (`validate_layered_1d_model/run_explicit_2d_hx_source.py`), which builds a
    domain around the survey and centres the source in it.

    The production model's extents deliberately are NOT reused: the production
    `sg.rss` carries the SEG-Y origin (`segy_ox`/`segy_oz`, e.g. a section
    starting several km down), whereas `build_homogeneous_grid` fixes the grid
    origin at 0. Writing the survey's absolute tx/rx coordinates into such a
    grid puts them outside it, and the engine then records all-zero traces with
    no error at all (see rockem-suite's aperture/geometry gotchas), which fits
    a calibration constant of exactly zero.

    Production x-offsets are kept. Receivers are placed both ABOVE and BELOW the
    transmitter by ``calibration_rx_dz_m`` so Hx-source Ey/Hz are nonzero even
    when the production survey is colinear (`rz0 == tz0`). Only the calibration
    FDTD/analytic pair uses this geometry; the production survey is unchanged.
    """
    dx = float(meta["dx_model_target_m"])
    nrx = max(1, int(meta.get("nrx", 1)))
    off_x_line = float(meta["rx0_m"]) + float(meta["drx_m"]) * np.arange(nrx, dtype=float)
    dz_cal = calibration_rx_dz_m(meta)

    max_off = float(np.max(np.abs(off_x_line))) if off_x_line.size else 0.0
    # `01_fw_setup` sizes apertx as 2*max_offset + aperture_margin; recover that
    # same margin and hold it clear of the PML on every side, in x and in depth.
    margin = max(float(meta.get("apertx_m", 0.0)) - 2.0 * max_off, MIN_PML_CLEARANCE_CELLS * dx)

    half_x = max_off + margin
    half_z = abs(dz_cal) + margin

    # Two lines at the same x-offsets: one above Tx, one below.
    off_x = np.concatenate([off_x_line, off_x_line])
    off_z = np.concatenate([
        np.full(off_x_line.shape, +dz_cal, dtype=float),
        np.full(off_x_line.shape, -dz_cal, dtype=float),
    ])
    return {
        "lx": 2.0 * half_x,
        "lz": 2.0 * half_z,
        "tx_x": half_x,
        "tx_z": half_z,
        "rx_x": half_x + off_x,
        "rx_z": half_z + off_z,
        "off_x": off_x,
        "off_z": off_z,
        "dz_cal_m": dz_cal,
        "nrx_line": int(nrx),
        "margin_m": margin,
    }


def homogeneous_analytic_gains(
    freqs_hz: Sequence[float],
    off_x: Sequence[float],
    tx_depth_m: float,
    rx_depth_m: float | Sequence[float],
    rho_ohm_m: float,
    eps_r: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Complex (Hx, Hz) gains [nfreq, nrx] for a homogeneous whole-space.

    ``rx_depth_m`` may be a scalar (all receivers at one depth) or a per-receiver
    depth array matching ``off_x``.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
    off_x = np.asarray(off_x, dtype=float).reshape(-1)
    rx_depths = np.asarray(rx_depth_m, dtype=float).reshape(-1)
    if rx_depths.size == 1:
        rx_depths = np.full(off_x.shape, float(rx_depths[0]), dtype=float)
    if rx_depths.shape != off_x.shape:
        raise ValueError(
            f"rx_depth_m size {rx_depths.size} does not match off_x size {off_x.size}"
        )

    layers = [Layer1D(float(rho_ohm_m), None, float(eps_r))]
    nfreq, nrx = freqs_hz.size, off_x.size
    hx = np.full((nfreq, nrx), np.nan, dtype=complex)
    hz = np.full((nfreq, nrx), np.nan, dtype=complex)
    for ifreq, f in enumerate(freqs_hz):
        for depth in np.unique(rx_depths):
            mask = rx_depths == depth
            _, hx_f, hz_f = magnetic_line_source_fields_layered(
                off_x[mask], float(f), layers, float(tx_depth_m), rx_depth_m=float(depth),
            )
            hx[ifreq, mask] = hx_f
            hz[ifreq, mask] = hz_f
    return hx, hz


def fit_global_C_per_frequency(
    fdtd_hx: np.ndarray,
    fdtd_hz: np.ndarray,
    analytic_hx: np.ndarray,
    analytic_hz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares C[f] pooling Hx and Hz so FDTD ≈ C * analytic."""
    fdtd_hx = np.asarray(fdtd_hx, dtype=complex)
    fdtd_hz = np.asarray(fdtd_hz, dtype=complex)
    analytic_hx = np.asarray(analytic_hx, dtype=complex)
    analytic_hz = np.asarray(analytic_hz, dtype=complex)
    nfreq = fdtd_hx.shape[0]
    c_out = np.full(nfreq, np.nan, dtype=complex)
    scatter_hx = np.full(nfreq, np.nan, dtype=float)
    scatter_hz = np.full(nfreq, np.nan, dtype=float)
    rel_scatter_hx = np.full(nfreq, np.nan, dtype=float)
    rel_scatter_hz = np.full(nfreq, np.nan, dtype=float)

    for f in range(nfreq):
        pred_hx, pred_hz = analytic_hx[f], analytic_hz[f]
        obs_hx, obs_hz = fdtd_hx[f], fdtd_hz[f]
        num = np.sum(np.conj(pred_hx) * obs_hx) + np.sum(np.conj(pred_hz) * obs_hz)
        den = np.sum(np.abs(pred_hx) ** 2) + np.sum(np.abs(pred_hz) ** 2)
        c_out[f] = num / den if den > 0 else np.nan + 0j

        def _scatter(obs, pred):
            resid = obs - c_out[f] * pred
            sc = float(np.sqrt(np.mean(np.abs(resid) ** 2)))
            ref = float(np.mean(np.abs(obs)))
            rel = sc / max(ref, 1e-300)
            return sc, rel

        scatter_hx[f], rel_scatter_hx[f] = _scatter(obs_hx, pred_hx)
        scatter_hz[f], rel_scatter_hz[f] = _scatter(obs_hz, pred_hz)

    return c_out, scatter_hx, scatter_hz, rel_scatter_hx, rel_scatter_hz


def sigmas_from_calibration(
    scatter_hx: np.ndarray,
    scatter_hz: np.ndarray,
    fdtd_hx: np.ndarray,
    fdtd_hz: np.ndarray,
    rel_floor: float = VALIDATED_REL_ERROR_FLOOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frequency sigma with validated relative floor."""
    fdtd_hx = np.asarray(fdtd_hx, dtype=complex)
    fdtd_hz = np.asarray(fdtd_hz, dtype=complex)
    floor_hx = rel_floor * np.mean(np.abs(fdtd_hx), axis=1)
    floor_hz = rel_floor * np.mean(np.abs(fdtd_hz), axis=1)
    sigma_hx = np.maximum(np.asarray(scatter_hx, dtype=float), floor_hx)
    sigma_hz = np.maximum(np.asarray(scatter_hz, dtype=float), floor_hz)
    return sigma_hx, sigma_hz


def compute_global_calibration_from_gains(
    fdtd_hx: np.ndarray,
    fdtd_hz: np.ndarray,
    analytic_hx: np.ndarray,
    analytic_hz: np.ndarray,
    freqs_hz: Sequence[float],
    *,
    rho_ohm_m: float,
    dx_m: float,
    method: str = "homogeneous_rho_min",
) -> dict:
    c_arr, sc_hx, sc_hz, rel_hx, rel_hz = fit_global_C_per_frequency(
        fdtd_hx, fdtd_hz, analytic_hx, analytic_hz,
    )
    sigma_hx, sigma_hz = sigmas_from_calibration(sc_hx, sc_hz, fdtd_hx, fdtd_hz)
    dx2 = float(dx_m) ** 2
    c_mag = np.abs(c_arr)
    return {
        "method": method,
        "rho_ohm_m": float(rho_ohm_m),
        "freqs_hz": [float(f) for f in np.asarray(freqs_hz, dtype=float).reshape(-1)],
        "C_hxhz_shared": c_arr,
        "sigma_hx": sigma_hx,
        "sigma_hz": sigma_hz,
        "scatter_hx_pct": (rel_hx * 100.0).tolist(),
        "scatter_hz_pct": (rel_hz * 100.0).tolist(),
        "C_magnitude": c_mag.tolist(),
        "C_phase_deg": np.angle(c_arr, deg=True).tolist(),
        "dx_m": float(dx_m),
        "dx_squared": dx2,
        "C_over_dx_squared": (c_mag / dx2).tolist(),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            f"Global homogeneous calibration at rho={rho_ohm_m:.4g} Ohm-m; "
            f"sigma floored at {rel_floor_pct(rel_floor=VALIDATED_REL_ERROR_FLOOR)}% of |FDTD|."
        ),
    }


def rel_floor_pct(*, rel_floor: float) -> float:
    return float(rel_floor * 100.0)


def prepare_homogeneous_calibration_run(
    fwd_2d_dir: Path | str,
    setup_meta_path: Path | str,
) -> dict:
    """Write homogeneous calibration FDTD inputs under calibration_homogeneous/.

    Returns the run directory together with the domain/survey it built, so the
    caller can report the geometry that the fit will actually be measured on.
    """
    fwd_2d_dir = Path(fwd_2d_dir)
    meta = _require_meta_fields(load_setup_metadata(setup_meta_path))
    run_dir = calibration_run_dir(fwd_2d_dir)
    data_dir = run_dir / "Data"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    rho_min = float(meta["rho_min_ohm_m"])
    eps_r = float(meta["eps_r_used"])
    dx = float(meta["dx_model_target_m"])
    dt = float(meta["dt_model_target_s"])
    lpml = int(meta.get("pml_heuristic", {}).get("lpml_cells", 13))
    pml = meta.get("pml_heuristic", {})
    apertx = float(meta.get("apertx_m", 200.0))
    fd_order = int(meta.get("fd_order", 2))

    geo = calibration_geometry(meta)
    grid = rockem_model.build_homogeneous_grid(
        resistivity_ohm_m=rho_min,
        domain_size_m=[geo["lx"], geo["lz"]],
        dx=dx,
        permittivity=eps_r,
        dim=2,
    )
    rockem_model.write_model_rss(grid, str(run_dir / "sg.rss"), str(run_dir / "ep.rss"))

    rockem_survey.write_survey_from_offsets(
        str(run_dir / "Survey.rss"),
        tx_x=geo["tx_x"], tx_z=geo["tx_z"], rx_x=geo["rx_x"], rx_z=geo["rx_z"], dim=2,
    )

    wav_src = fwd_2d_dir / "wav2d.rss"
    if not wav_src.exists():
        raise FileNotFoundError(f"Production wavelet not found: {wav_src}")
    shutil.copy2(wav_src, run_dir / "wav2d.rss")

    rockem_config.write_te2d_config(
        str(run_dir / CALIBRATION_CFG_NAME),
        sg_file="sg.rss",
        ep_file="ep.rss",
        wavelet_file="wav2d.rss",
        survey_file="Survey.rss",
        source_field="HX",
        order=fd_order,
        lpml=lpml,
        adi=False,
        usepml=True,
        snapinc=1000,
        dtrec=dt,
        apertx=apertx,
        pml_kmax=float(pml.get("pml_kmax", -1.0)),
        pml_smax=float(pml.get("pml_smax", -1.0)),
        pml_amax=float(pml.get("pml_amax", -1.0)),
        records=("EY", "HX", "HZ"),
        recordfiles={"HX": "Data/Hxshot.rss", "HZ": "Data/Hzshot.rss", "EY": "Data/Eyshot.rss"},
    )
    nx, _, nz = grid.conductivity.shape
    return {"run_dir": run_dir, "nx": int(nx), "nz": int(nz), **geo}


def compute_calibration_from_fdtd_outputs(
    fwd_2d_dir: Path | str,
    setup_meta_path: Path | str,
    freqs_hz: Sequence[float],
    *,
    f_min_hz: float,
    n_periods_extract: float = 3.0,
) -> dict:
    """Extract FDTD gains from calibration run, fit C against homogeneous analytic."""
    from scripts.modules.fd_visualization import compute_gains_for_fd_outputs

    meta = _require_meta_fields(load_setup_metadata(setup_meta_path))
    run_dir = calibration_run_dir(fwd_2d_dir)
    hx_path = run_dir / "Data" / "Hxshot.rss"
    hz_path = run_dir / "Data" / "Hzshot.rss"
    wav_path = run_dir / "wav2d.rss"
    for p in (hx_path, hz_path, wav_path):
        if not p.exists():
            raise FileNotFoundError(f"Calibration output missing: {p} (run calibration FDTD first).")

    fdtd = compute_gains_for_fd_outputs(
        hx_path, hz_path, wav_path,
        freqs=np.asarray(freqs_hz, dtype=float),
        f_min_hz=float(f_min_hz),
        n_periods_extract=float(n_periods_extract),
    )
    geo = fdtd["geometry"]
    tx_x = float(np.asarray(geo["src_x"], dtype=float)[0])
    tx_z = float(np.asarray(geo["src_z"], dtype=float)[0])
    rx_x = np.asarray(geo["rx_x"], dtype=float)
    rx_z = np.asarray(geo["rx_z"], dtype=float)
    off_x = rx_x - tx_x

    rho_min = float(meta["rho_min_ohm_m"])
    eps_r = float(meta["eps_r_used"])
    dx = float(meta["dx_model_target_m"])
    analytic_hx, analytic_hz = homogeneous_analytic_gains(
        freqs_hz, off_x, tx_z, rx_z, rho_min, eps_r,
    )
    fdtd_hx = np.asarray(fdtd["Hx"]["gain"], dtype=complex)
    fdtd_hz = np.asarray(fdtd["Hz"]["gain"], dtype=complex)
    if not np.any(np.abs(fdtd_hx) > 0.0) and not np.any(np.abs(fdtd_hz) > 0.0):
        raise ValueError(
            "Calibration FDTD recorded all-zero traces, so no calibration can be fitted "
            "(this would otherwise be reported as |C| = 0 at every frequency). The usual "
            "cause is a source or receiver outside the modelled grid, which the engine "
            "records as zeros without raising. Check the calibration survey against the "
            f"domain written in {run_dir}."
        )
    if not np.any(np.abs(analytic_hz) > 0.0):
        raise ValueError(
            "Homogeneous analytic Hz is identically zero on the calibration survey, so Hz "
            "cannot enter C. Receivers must sit above/below the transmitter "
            f"(expected |dz| >= {calibration_rx_dz_m(meta):g} m)."
        )

    cal = compute_global_calibration_from_gains(
        fdtd_hx, fdtd_hz, analytic_hx, analytic_hz, freqs_hz,
        rho_ohm_m=rho_min, dx_m=dx,
    )
    cal["fdtd_result"] = fdtd
    cal["analytic_hx"] = analytic_hx
    cal["analytic_hz"] = analytic_hz
    cal["calibration_rx_dz_m"] = float(calibration_rx_dz_m(meta))
    cal["notes"] = (
        f"{cal.get('notes', '')} Calibration receivers at ±{cal['calibration_rx_dz_m']:.4g} m "
        "relative to Tx (Hx and Hz both enter C)."
    ).strip()
    return cal


def _json_safe_calibration_payload(cal: Mapping[str, Any]) -> dict:
    payload = {k: v for k, v in cal.items() if k not in ("fdtd_result", "analytic_hx", "analytic_hz")}
    c_arr = np.asarray(payload.pop("C_hxhz_shared"), dtype=complex)
    payload["C_hxhz_shared_real"] = np.real(c_arr).tolist()
    payload["C_hxhz_shared_imag"] = np.imag(c_arr).tolist()
    for key, val in list(payload.items()):
        if isinstance(val, np.ndarray):
            payload[key] = val.tolist()
    return payload


def save_calibration_to_metadata(setup_meta_path: Path | str, cal: Mapping[str, Any]) -> dict:
    path = Path(setup_meta_path)
    meta = load_setup_metadata(path) if path.exists() else {}
    meta["fdtd_analytic_calibration"] = _json_safe_calibration_payload(cal)
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_global_calibration(setup_meta_path: Path | str) -> dict:
    meta = load_setup_metadata(setup_meta_path)
    block = meta.get("fdtd_analytic_calibration")
    if block is None:
        raise KeyError(
            "No fdtd_analytic_calibration in setup_metadata.json — press 'Calibrate' "
            "in notebook 02 to run the homogeneous calibration and store C."
        )
    out = dict(block)
    if "C_hxhz_shared" not in out:
        re = np.asarray(out["C_hxhz_shared_real"], dtype=float)
        im = np.asarray(out["C_hxhz_shared_imag"], dtype=float)
        out["C_hxhz_shared"] = re + 1j * im
    else:
        out["C_hxhz_shared"] = np.asarray(out["C_hxhz_shared"], dtype=complex)
    out["freqs_hz"] = np.asarray(out["freqs_hz"], dtype=float)
    out["sigma_hx"] = np.asarray(out.get("sigma_hx", []), dtype=float)
    out["sigma_hz"] = np.asarray(out.get("sigma_hz", []), dtype=float)
    return out


def apply_calibration_to_gains(hx: np.ndarray, hz: np.ndarray, c_per_freq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (C*hx, C*hz) with C broadcast over receivers."""
    c = np.asarray(c_per_freq, dtype=complex).reshape(-1, 1)
    return c * np.asarray(hx, dtype=complex), c * np.asarray(hz, dtype=complex)


def calibration_for_inversion(setup_meta_path: Path | str) -> dict:
    """Format loaded global calibration for notebook 05 inversion cfg."""
    cal = load_global_calibration(setup_meta_path)
    return {
        "C": np.asarray(cal["C_hxhz_shared"], dtype=complex),
        "sigma_hx": np.asarray(cal["sigma_hx"], dtype=float),
        "sigma_hz": np.asarray(cal["sigma_hz"], dtype=float),
        "freqs_hz": np.asarray(cal["freqs_hz"], dtype=float),
        "method": cal.get("method", "homogeneous_rho_min"),
        "notes": cal.get("notes", ""),
    }


__all__ = [
    "VALIDATED_REL_ERROR_FLOOR",
    "CALIBRATION_SUBDIR",
    "CALIBRATION_CFG_NAME",
    "CALIBRATION_RX_DZ_M",
    "MIN_CALIBRATION_RX_DZ_CELLS",
    "MIN_PML_CLEARANCE_CELLS",
    "apply_calibration_to_gains",
    "calibration_for_inversion",
    "calibration_geometry",
    "calibration_rx_dz_m",
    "calibration_run_dir",
    "compute_calibration_from_fdtd_outputs",
    "compute_global_calibration_from_gains",
    "fit_global_C_per_frequency",
    "homogeneous_analytic_gains",
    "load_global_calibration",
    "load_setup_metadata",
    "prepare_homogeneous_calibration_run",
    "save_calibration_to_metadata",
]
