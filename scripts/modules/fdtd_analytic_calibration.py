"""Global FDTD-vs-analytic calibration for the workshop.

Computes a single Tx-independent complex scale C[f] per frequency such that
FDTD channel gain ≈ C * analytic channel gain, following rockem-suite's
convention (see validate_layered_1d_model/README.md: C = FDTD/analytic,
|C| ≈ dx²).

Two Earth models are supported (notebook 02 buttons; last run wins in
setup_metadata):

1. Homogeneous ``rho_min`` — receivers at ±depth so Hx-source Hz is nonzero.
2. Lateral average of production ``sg.rss`` resistivity (vertically varying 1D)
   with the production survey offsets / apertx from Step 01.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scripts.modules.analytic_1d_forward import ForwardRejected, Layer1D
from scripts.modules.rockem_bridge import (
    GreensSolverError,
    config as rockem_config,
    magnetic_line_source_fields_layered,
    model as rockem_model,
    survey as rockem_survey,
)
from third_party.rockseis.io.rsfile import rsfile

VALIDATED_REL_ERROR_FLOOR = 0.03
CALIBRATION_SUBDIR = "calibration_homogeneous"
CALIBRATION_SUBDIR_LATERAL = "calibration_lateral_average"
CALIBRATION_CFG_NAME = "mod_cal.cfg"
METHOD_HOMOGENEOUS = "homogeneous_rho_min"
METHOD_LATERAL_AVERAGE = "lateral_average_true"
# Keep sources/receivers this many cells clear of the PML when the survey's own
# aperture margin is unavailable (rockem-suite gotchas: 8-16 cells).
MIN_PML_CLEARANCE_CELLS = 16
# rockem-suite Hx-source validation uses rx_dz = -20 m so Ey/Hz are nonzero in a
# homogeneous medium (they null exactly at the source depth). The homogeneous
# calibration survey places receivers both above and below the Tx by at least
# this amount (and at least two grid cells), independent of production rz0/tz0.
# Lateral-average calibration keeps production rz0−tz0 instead.
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


def calibration_run_dir(fwd_2d_dir: Path | str, method: str = METHOD_HOMOGENEOUS) -> Path:
    sub = CALIBRATION_SUBDIR_LATERAL if method == METHOD_LATERAL_AVERAGE else CALIBRATION_SUBDIR
    return Path(fwd_2d_dir) / sub


def calibration_rx_dz_m(meta: Mapping[str, Any]) -> float:
    """Vertical Tx–Rx separation used only for the homogeneous calibration survey."""
    dx = float(meta["dx_model_target_m"])
    return float(max(CALIBRATION_RX_DZ_M, MIN_CALIBRATION_RX_DZ_CELLS * dx))


def _aperture_margin_m(meta: Mapping[str, Any], max_off_x: float) -> float:
    """Recover Step 01 aperture margin: apertx ≈ 2*max_offset + margin."""
    dx = float(meta["dx_model_target_m"])
    return float(
        max(float(meta.get("apertx_m", 0.0)) - 2.0 * float(max_off_x), MIN_PML_CLEARANCE_CELLS * dx)
    )


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
    margin = _aperture_margin_m(meta, max_off)

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


def calibration_geometry_production(meta: Mapping[str, Any]) -> dict:
    """Source-centred domain using production survey offsets from Step 01.

    Same apertx-derived margin as homogeneous calibration, but ``off_z`` is
    ``rz0_m - tz0_m`` (one receiver depth), matching the production survey.

    Receivers that coincide with the source (offset ~ 0 and same depth) are
    dropped — the line-source field is singular there and cannot enter ``C``.
    """
    nrx = max(1, int(meta.get("nrx", 1)))
    off_x = float(meta["rx0_m"]) + float(meta["drx_m"]) * np.arange(nrx, dtype=float)
    off_z_val = float(meta["rz0_m"]) - float(meta["tz0_m"])
    off_z = np.full(off_x.shape, off_z_val, dtype=float)

    # Analytic Green's function is singular at the source point.
    dx = float(meta["dx_model_target_m"])
    singular = (np.abs(off_x) < 0.1 * dx) & (np.abs(off_z) < 0.1 * dx)
    if np.any(singular):
        off_x = off_x[~singular]
        off_z = off_z[~singular]
    if off_x.size < 1:
        raise ValueError(
            "Production survey has no usable receivers for lateral-average calibration "
            "(all receivers coincide with the source). Adjust rx0/drx/nrx or rz0≠tz0."
        )

    max_off = float(np.max(np.abs(off_x)))
    margin = _aperture_margin_m(meta, max_off)
    half_x = max_off + margin
    half_z = abs(off_z_val) + margin

    return {
        "lx": 2.0 * half_x,
        "lz": 2.0 * half_z,
        "tx_x": half_x,
        "tx_z": half_z,
        "rx_x": half_x + off_x,
        "rx_z": half_z + off_z,
        "off_x": off_x,
        "off_z": off_z,
        "dz_prod_m": off_z_val,
        "nrx_line": int(off_x.size),
        "margin_m": margin,
        "tz0_prod_m": float(meta["tz0_m"]),
    }


def _read_rss_conductivity_xz(path: Path | str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_centers, z_centers, sigma[nx,nz]) from an RSS conductivity model."""
    path = Path(path)
    f = rsfile()
    f.read(str(path))
    data = np.asarray(f.data, dtype=float)
    if data.ndim == 3:
        iy = int(data.shape[1] // 2)
        data_xz = np.asarray(data[:, iy, :], dtype=float)
    else:
        data_xz = np.squeeze(data)
        if data_xz.ndim == 3:
            iy = int(data_xz.shape[1] // 2)
            data_xz = np.asarray(data_xz[:, iy, :], dtype=float)
    if data_xz.ndim != 2:
        raise ValueError(f"Expected 2D/3D conductivity RSS at {path}, got shape {data.shape}")

    nx, nz = int(data_xz.shape[0]), int(data_xz.shape[1])
    dx = float(f.geomD[0]) if f.geomD[0] else 1.0
    ox = float(f.geomO[0])
    iz = 2 if (len(f.geomN) > 2 and int(f.geomN[2]) > 0) else 1
    dz = float(f.geomD[iz]) if f.geomD[iz] else dx
    oz = float(f.geomO[iz])
    x = ox + (np.arange(nx, dtype=float) + 0.5) * dx
    z = oz + (np.arange(nz, dtype=float) + 0.5) * dz
    return x, z, np.asarray(data_xz, dtype=float)


def lateral_average_resistivity_profile(sg_path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """Vertically varying 1D resistivity: mean of ρ across x at each z."""
    _x, z, sigma = _read_rss_conductivity_xz(sg_path)
    sigma = np.clip(np.asarray(sigma, dtype=float), 1e-12, None)
    rho = 1.0 / sigma
    rho_1d = np.mean(rho, axis=0)
    return np.asarray(z, dtype=float), np.asarray(rho_1d, dtype=float)


def _blocky_layers_from_rho_trace(
    rho: np.ndarray,
    dz: float,
    eps_r: float,
) -> List[Any]:
    """Merge adjacent equal-ρ cells into rockem LayerSpec list (last = halfspace)."""
    rho = np.asarray(rho, dtype=float).reshape(-1)
    if rho.size == 0:
        raise ValueError("Empty resistivity profile for lateral-average calibration.")
    dz = float(dz)
    if dz <= 0.0:
        raise ValueError("dz must be positive.")

    specs: List[Any] = []
    i = 0
    while i < rho.size:
        j = i + 1
        while j < rho.size and abs(rho[j] - rho[i]) <= 1e-12 * max(1.0, abs(rho[i])):
            j += 1
        n_cells = j - i
        if j >= rho.size:
            # Last run becomes the halfspace (extends to domain edges).
            specs.append(
                rockem_model.LayerSpec(
                    resistivity_ohm_m=float(rho[i]),
                    thickness_m=None,
                    permittivity=float(eps_r),
                )
            )
        else:
            specs.append(
                rockem_model.LayerSpec(
                    resistivity_ohm_m=float(rho[i]),
                    thickness_m=float(n_cells * dz),
                    permittivity=float(eps_r),
                )
            )
        i = j

    if not specs:
        raise ValueError("Failed to build layers from lateral-average profile.")
    if specs[-1].thickness_m is not None:
        # Single-cell domain or all merged before last: force halfspace.
        last = specs[-1]
        specs[-1] = rockem_model.LayerSpec(
            resistivity_ohm_m=float(last.resistivity_ohm_m),
            thickness_m=None,
            permittivity=float(eps_r),
        )
    return specs


def layers_from_lateral_average_for_window(
    z_prod: np.ndarray,
    rho_1d: np.ndarray,
    tz0_prod_m: float,
    half_z_m: float,
    dx_m: float,
    eps_r: float,
) -> List[Any]:
    """Sample ρ̄ on the Tx-centred depth window and merge to LayerSpecs."""
    z_prod = np.asarray(z_prod, dtype=float).reshape(-1)
    rho_1d = np.asarray(rho_1d, dtype=float).reshape(-1)
    if z_prod.size != rho_1d.size:
        raise ValueError("z_prod and rho_1d size mismatch.")
    if z_prod.size < 2:
        raise ValueError("Need at least two depth samples in production sg.rss.")

    # Production z may increase downward or upward; sort ascending for interp.
    order = np.argsort(z_prod)
    z_s = z_prod[order]
    rho_s = rho_1d[order]
    dz_prod = float(np.median(np.diff(z_s)))
    if dz_prod <= 0.0:
        raise ValueError("Production depth axis is not increasing.")

    dz = float(dx_m)  # isotropic cal grid (same as homogeneous path)
    n_half = max(1, int(np.ceil(float(half_z_m) / dz)))
    # Cells from -n_half*dz … +(n_half-1)*dz relative to Tx → span 2*n_half*dz
    # so finite+halfspace centering covers ~lz.
    z_rel = (np.arange(-n_half, n_half, dtype=float) + 0.5) * dz
    z_query = float(tz0_prod_m) + z_rel
    rho_win = np.interp(z_query, z_s, rho_s, left=float(rho_s[0]), right=float(rho_s[-1]))
    return _blocky_layers_from_rho_trace(rho_win, dz=dz, eps_r=eps_r)


def layered_analytic_gains(
    freqs_hz: Sequence[float],
    off_x: Sequence[float],
    tx_depth_m: float,
    rx_depth_m: float | Sequence[float],
    layers: Sequence[Layer1D],
) -> Tuple[np.ndarray, np.ndarray]:
    """Complex (Hx, Hz) gains for a layered stack (same layers as FDTD)."""
    freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
    off_x = np.asarray(off_x, dtype=float).reshape(-1)
    rx_depths = np.asarray(rx_depth_m, dtype=float).reshape(-1)
    if rx_depths.size == 1:
        rx_depths = np.full(off_x.shape, float(rx_depths[0]), dtype=float)
    if rx_depths.shape != off_x.shape:
        raise ValueError(
            f"rx_depth_m size {rx_depths.size} does not match off_x size {off_x.size}"
        )

    nfreq, nrx = freqs_hz.size, off_x.size
    hx = np.full((nfreq, nrx), np.nan, dtype=complex)
    hz = np.full((nfreq, nrx), np.nan, dtype=complex)
    layer_list = list(layers)
    for ifreq, f in enumerate(freqs_hz):
        for depth in np.unique(rx_depths):
            mask = rx_depths == depth
            try:
                _, hx_f, hz_f = magnetic_line_source_fields_layered(
                    off_x[mask], float(f), layer_list, float(tx_depth_m), rx_depth_m=float(depth),
                )
            except GreensSolverError as exc:
                raise ForwardRejected(
                    f"Lateral-average analytic forward rejected ({exc})."
                ) from exc
            hx[ifreq, mask] = hx_f
            hz[ifreq, mask] = hz_f
    if not (np.all(np.isfinite(hx)) and np.all(np.isfinite(hz))):
        raise ForwardRejected("non-finite layered analytic calibration forward")
    return hx, hz


def _layer1d_from_rockem_specs(specs: Sequence[Any]) -> List[Layer1D]:
    out: List[Layer1D] = []
    for s in specs:
        out.append(
            Layer1D(
                float(s.resistivity_ohm_m),
                None if s.thickness_m is None else float(s.thickness_m),
                float(s.permittivity),
            )
        )
    return out



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
    method: str = METHOD_HOMOGENEOUS,
    notes: Optional[str] = None,
) -> dict:
    c_arr, sc_hx, sc_hz, rel_hx, rel_hz = fit_global_C_per_frequency(
        fdtd_hx, fdtd_hz, analytic_hx, analytic_hz,
    )
    sigma_hx, sigma_hz = sigmas_from_calibration(sc_hx, sc_hz, fdtd_hx, fdtd_hz)
    dx2 = float(dx_m) ** 2
    c_mag = np.abs(c_arr)
    if notes is None:
        notes = (
            f"Global calibration (method={method}) at rho_ref={rho_ohm_m:.4g} Ohm-m; "
            f"sigma floored at {rel_floor_pct(rel_floor=VALIDATED_REL_ERROR_FLOOR)}% of |FDTD|."
        )
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
        "notes": notes,
    }


def rel_floor_pct(*, rel_floor: float) -> float:
    return float(rel_floor * 100.0)


def _write_calibration_survey_and_cfg(
    *,
    run_dir: Path,
    geo: Mapping[str, Any],
    grid: Any,
    meta: Mapping[str, Any],
    fwd_2d_dir: Path,
) -> None:
    data_dir = run_dir / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rockem_model.write_model_rss(grid, str(run_dir / "sg.rss"), str(run_dir / "ep.rss"))
    rockem_survey.write_survey_from_offsets(
        str(run_dir / "Survey.rss"),
        tx_x=geo["tx_x"], tx_z=geo["tx_z"], rx_x=geo["rx_x"], rx_z=geo["rx_z"], dim=2,
    )
    wav_src = fwd_2d_dir / "wav2d.rss"
    if not wav_src.exists():
        raise FileNotFoundError(f"Production wavelet not found: {wav_src}")
    shutil.copy2(wav_src, run_dir / "wav2d.rss")

    dt = float(meta["dt_model_target_s"])
    lpml = int(meta.get("pml_heuristic", {}).get("lpml_cells", 13))
    pml = meta.get("pml_heuristic", {})
    apertx = float(meta.get("apertx_m", 200.0))
    fd_order = int(meta.get("fd_order", 2))
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
    run_dir = calibration_run_dir(fwd_2d_dir, METHOD_HOMOGENEOUS)
    run_dir.mkdir(parents=True, exist_ok=True)

    rho_min = float(meta["rho_min_ohm_m"])
    eps_r = float(meta["eps_r_used"])
    dx = float(meta["dx_model_target_m"])

    geo = calibration_geometry(meta)
    grid = rockem_model.build_homogeneous_grid(
        resistivity_ohm_m=rho_min,
        domain_size_m=[geo["lx"], geo["lz"]],
        dx=dx,
        permittivity=eps_r,
        dim=2,
    )
    _write_calibration_survey_and_cfg(
        run_dir=run_dir, geo=geo, grid=grid, meta=meta, fwd_2d_dir=fwd_2d_dir,
    )
    nx, _, nz = grid.conductivity.shape
    return {"run_dir": run_dir, "nx": int(nx), "nz": int(nz), "method": METHOD_HOMOGENEOUS, **geo}


def prepare_lateral_average_calibration_run(
    fwd_2d_dir: Path | str,
    setup_meta_path: Path | str,
) -> dict:
    """Write 1D lateral-average calibration inputs under calibration_lateral_average/.

    Earth is the x-average of production ``sg.rss`` resistivity (vertically
    varying 1D). Survey offsets and apertx/dx/dt/eps/pml come from Step 01 meta.
    """
    fwd_2d_dir = Path(fwd_2d_dir)
    meta = _require_meta_fields(load_setup_metadata(setup_meta_path))
    sg_prod = fwd_2d_dir / "sg.rss"
    if not sg_prod.exists():
        raise FileNotFoundError(
            f"Production model not found: {sg_prod}. Run Step 01 before lateral-average calibration."
        )

    run_dir = calibration_run_dir(fwd_2d_dir, METHOD_LATERAL_AVERAGE)
    run_dir.mkdir(parents=True, exist_ok=True)

    eps_r = float(meta["eps_r_used"])
    dx = float(meta["dx_model_target_m"])
    geo = calibration_geometry_production(meta)
    z_prod, rho_1d = lateral_average_resistivity_profile(sg_prod)
    half_z = 0.5 * float(geo["lz"])
    specs = layers_from_lateral_average_for_window(
        z_prod, rho_1d, float(geo["tz0_prod_m"]), half_z, dx, eps_r,
    )
    grid = rockem_model.build_layered_1d_grid(
        layers=specs,
        domain_size_m=[geo["lx"], geo["lz"]],
        dx=dx,
        tx_depth_m=float(geo["tx_z"]),
        dim=2,
    )
    _write_calibration_survey_and_cfg(
        run_dir=run_dir, geo=geo, grid=grid, meta=meta, fwd_2d_dir=fwd_2d_dir,
    )
    # Persist layer specs for the fit path (JSON-friendly).
    layer_payload = [
        {
            "resistivity_ohm_m": float(s.resistivity_ohm_m),
            "thickness_m": None if s.thickness_m is None else float(s.thickness_m),
            "permittivity": float(s.permittivity),
        }
        for s in specs
    ]
    (run_dir / "lateral_average_layers.json").write_text(
        json.dumps({"layers": layer_payload, "tz0_prod_m": float(geo["tz0_prod_m"])}, indent=2) + "\n"
    )
    nx, _, nz = grid.conductivity.shape
    rho_mean = float(np.mean(rho_1d))
    return {
        "run_dir": run_dir,
        "nx": int(nx),
        "nz": int(nz),
        "method": METHOD_LATERAL_AVERAGE,
        "layers": specs,
        "rho_mean_ohm_m": rho_mean,
        **geo,
    }


def compute_calibration_from_fdtd_outputs(
    fwd_2d_dir: Path | str,
    setup_meta_path: Path | str,
    freqs_hz: Sequence[float],
    *,
    f_min_hz: float,
    n_periods_extract: float = 3.0,
    method: str = METHOD_HOMOGENEOUS,
) -> dict:
    """Extract FDTD gains from a calibration run and fit global C vs analytic."""
    from scripts.modules.fd_visualization import compute_gains_for_fd_outputs

    method = str(method)
    if method not in (METHOD_HOMOGENEOUS, METHOD_LATERAL_AVERAGE):
        raise ValueError(f"Unknown calibration method: {method}")

    meta = _require_meta_fields(load_setup_metadata(setup_meta_path))
    run_dir = calibration_run_dir(fwd_2d_dir, method)
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
    tx_z = float(np.asarray(geo["src_z"], dtype=float)[0])
    rx_x = np.asarray(geo["rx_x"], dtype=float)
    rx_z = np.asarray(geo["rx_z"], dtype=float)
    tx_x = float(np.asarray(geo["src_x"], dtype=float)[0])
    off_x = rx_x - tx_x

    eps_r = float(meta["eps_r_used"])
    dx = float(meta["dx_model_target_m"])
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

    if method == METHOD_HOMOGENEOUS:
        rho_ref = float(meta["rho_min_ohm_m"])
        analytic_hx, analytic_hz = homogeneous_analytic_gains(
            freqs_hz, off_x, tx_z, rx_z, rho_ref, eps_r,
        )
        if not np.any(np.abs(analytic_hz) > 0.0):
            raise ValueError(
                "Homogeneous analytic Hz is identically zero on the calibration survey, so Hz "
                "cannot enter C. Receivers must sit above/below the transmitter "
                f"(expected |dz| >= {calibration_rx_dz_m(meta):g} m)."
            )
        notes = (
            f"Global homogeneous calibration at rho={rho_ref:.4g} Ohm-m; "
            f"sigma floored at {rel_floor_pct(rel_floor=VALIDATED_REL_ERROR_FLOOR)}% of |FDTD|. "
            f"Calibration receivers at ±{calibration_rx_dz_m(meta):.4g} m relative to Tx."
        )
    else:
        layers_path = run_dir / "lateral_average_layers.json"
        if not layers_path.exists():
            raise FileNotFoundError(
                f"Missing {layers_path.name} — re-run prepare_lateral_average_calibration_run."
            )
        payload = json.loads(layers_path.read_text())
        specs = [
            rockem_model.LayerSpec(
                resistivity_ohm_m=float(L["resistivity_ohm_m"]),
                thickness_m=None if L.get("thickness_m") is None else float(L["thickness_m"]),
                permittivity=float(L.get("permittivity", eps_r)),
            )
            for L in payload["layers"]
        ]
        layer1d = _layer1d_from_rockem_specs(specs)
        try:
            analytic_hx, analytic_hz = layered_analytic_gains(
                freqs_hz, off_x, tx_z, rx_z, layer1d,
            )
        except ForwardRejected as exc:
            raise ValueError(str(exc)) from exc
        rho_ref = float(np.mean([float(s.resistivity_ohm_m) for s in specs]))
        dz_prod = float(meta["rz0_m"]) - float(meta["tz0_m"])
        notes = (
            f"Global lateral-average 1D calibration from production sg.rss "
            f"(mean rho≈{rho_ref:.4g} Ohm-m, {len(specs)} layers); "
            f"production survey offsets (rz0-tz0={dz_prod:.4g} m); "
            f"sigma floored at {rel_floor_pct(rel_floor=VALIDATED_REL_ERROR_FLOOR)}% of |FDTD|."
        )

    cal = compute_global_calibration_from_gains(
        fdtd_hx, fdtd_hz, analytic_hx, analytic_hz, freqs_hz,
        rho_ohm_m=rho_ref, dx_m=dx, method=method, notes=notes,
    )
    cal["fdtd_result"] = fdtd
    cal["analytic_hx"] = analytic_hx
    cal["analytic_hz"] = analytic_hz
    if method == METHOD_HOMOGENEOUS:
        cal["calibration_rx_dz_m"] = float(calibration_rx_dz_m(meta))
    else:
        cal["dz_prod_m"] = float(meta["rz0_m"]) - float(meta["tz0_m"])
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
            "No fdtd_analytic_calibration in setup_metadata.json — press a Calibrate "
            "button in notebook 02 (homogeneous or lateral-average) to store C."
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
    "CALIBRATION_SUBDIR_LATERAL",
    "CALIBRATION_CFG_NAME",
    "CALIBRATION_RX_DZ_M",
    "MIN_CALIBRATION_RX_DZ_CELLS",
    "MIN_PML_CLEARANCE_CELLS",
    "METHOD_HOMOGENEOUS",
    "METHOD_LATERAL_AVERAGE",
    "apply_calibration_to_gains",
    "calibration_for_inversion",
    "calibration_geometry",
    "calibration_geometry_production",
    "calibration_rx_dz_m",
    "calibration_run_dir",
    "compute_calibration_from_fdtd_outputs",
    "compute_global_calibration_from_gains",
    "fit_global_C_per_frequency",
    "homogeneous_analytic_gains",
    "lateral_average_resistivity_profile",
    "layered_analytic_gains",
    "layers_from_lateral_average_for_window",
    "load_global_calibration",
    "load_setup_metadata",
    "prepare_homogeneous_calibration_run",
    "prepare_lateral_average_calibration_run",
    "save_calibration_to_metadata",
]
