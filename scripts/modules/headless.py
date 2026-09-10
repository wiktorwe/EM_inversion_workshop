"""Headless (non-GUI) drivers for the workshop's Step 01/02 pipeline.

The six workshop notebooks are Voila GUIs; every numerical step they perform
lives in ``scripts.modules.*`` but the *orchestration* (load SEG-Y -> design FD
-> resample -> write cfg -> run engine -> calibrate) only exists inside the
notebooks' button handlers. This module re-implements that orchestration as
plain functions so the same pipeline can be scripted: A/B experiments (FD
stencil order, per-frequency vs broadband runs), timing measurements, and
regression checks.

It is deliberately a *driver*, not a second implementation: every numerical
decision is delegated to the same helpers the notebooks call
(``design_explicit_fd``, ``write_sg_ep_rss``, ``create_wavelet_rss``,
``generate_survey_rss``, ``update_modcfg_for_workshop``,
``compute_calibration_from_fdtd_outputs``). If a notebook and this module ever
disagree, this module is wrong.

Every numeric switch announces itself: ``build_forward_inputs`` prints the FD
design line (order, kappa, dx, dt, eps_r, nt) it actually wrote, and
``run_forward`` prints the ``order``/``lpml``/``source_type`` it read back out
of the cfg on disk immediately before launching the engine.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from scripts.modules.workshop_config import load_config
from scripts.modules.fd import (
    C0,
    ExplicitDesignInputs,
    design_explicit_fd,
    enforce_rss_min_value,
    interpolate_rss_python,
    read_cfg_values,
    resolve_fd_order_from_cfg,
    update_cfg_values,
    update_modcfg_for_workshop,
)
from scripts.modules.segy import (
    pad_resistivity_for_depth_margin,
    read_resistivity_from_segy,
    write_sg_ep_rss,
)
from scripts.modules.source import create_wavelet_rss
from scripts.modules.survey import (
    generate_survey_rss,
    update_cfg_values as update_survey_cfg,
)

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = ROOT / "scripts" / "templates"

# Source-field name -> the TE2D engine's numeric source_type (mod.cfg / inv.cfg
# "1=EY 3=HX 5=HZ"; see rockem-suite gotchas - the codes are per-engine and
# non-contiguous, so never hardcode the digit at a call site).
SOURCE_TYPE_CODES = {"EY": 1, "HX": 3, "HZ": 5}


@dataclass
class SetupParams:
    """Every Step 01 widget value, as a plain dataclass.

    Defaults reproduce the workshop's stored production setup (the one behind
    ``workspace/2D/forward/setup_metadata.json``): 30 transmitters along a
    horizontal well at z = 6050 m, two colinear receivers at -13.1 and -25.3 m
    offset, and a four-tone 1/2/4/6 kHz wavelet.
    """

    segy_path: Path = ROOT / "examples" / "Fault_1.sgy"

    # Section 2 - wavelet
    flist_hz: Sequence[float] = (1000.0, 2000.0, 4000.0, 6000.0)
    wavelet_dt_s: float = 1e-5
    n_periods: int = 5
    alpha: float = 0.5

    # Section 3 - survey
    tx0_m: float = 1400.0
    tz0_m: float = 6050.0
    dtx_m: float = 4.8
    ntx: int = 30
    rx0_m: float = -13.1
    rz0_m: float = 6050.0
    drx_m: float = -12.2
    nrx: int = 2

    # Section 4 - FD design
    f_min_hz: Optional[float] = None   # None -> min(flist_hz)
    f_max_hz: Optional[float] = None   # None -> max(flist_hz)
    rho_min_ohm_m: Optional[float] = 1.0     # None -> from the loaded model
    rho_max_ohm_m: Optional[float] = 150.0   # None -> from the loaded model
    points_per_skin: int = 8
    cells_per_min_offset: int = 8
    tan_delta_floor: float = 50.0
    eps_r_cap: float = 5000.0          # see fd.ExplicitDesignInputs.eps_r_cap
    sigma_max_clip_s_per_m: Optional[float] = None

    # Aperture override. `design_explicit_fd` sizes apertx from the SURVEY
    # OFFSETS (2*max_offset + margin), which is right for the direct couplings
    # and far too small for a look-ahead study: apertx is a source-centred TOTAL
    # width, so structure further than apertx/2 from a transmitter is not in that
    # shot's local model at all and cannot be detected. Set this from the
    # look-ahead range you want to resolve instead.
    apertx_override_m: Optional[float] = None

    # FD stencil / engine
    fd_order: Optional[int] = None     # None -> read from the mod.cfg template
    lpml: Optional[int] = None         # None -> read from the mod.cfg template
    source_field: str = "HX"           # "HX" or "HZ" (Task 5's second source)

    def resolved_f_min(self) -> float:
        return float(self.f_min_hz if self.f_min_hz is not None else min(self.flist_hz))

    def resolved_f_max(self) -> float:
        return float(self.f_max_hz if self.f_max_hz is not None else max(self.flist_hz))


def _offset_bounds(p: SetupParams) -> tuple[float, float]:
    offs = [abs(p.rx0_m + i * p.drx_m) for i in range(int(p.nrx))]
    offs = [o for o in offs if o > 0.0]
    if not offs:
        raise ValueError("Receiver offsets are all zero - cannot size dx/apertx from them.")
    return min(offs), max(offs)


def _template_cfg_int(key: str, default: int) -> int:
    values = read_cfg_values(TEMPLATES_DIR / "mod.cfg")
    return int(float(values.get(key, default)))


def fd_design_for(p: SetupParams, *, max_depth_offset_m: float) -> tuple[dict, int, int]:
    """Run ``design_explicit_fd`` for these params; return (design, order, lpml).

    ``fd_order``/``lpml`` default to whatever ``scripts/templates/mod.cfg``
    says, which is the chain Task 1 depends on: the cfg is the single place the
    stencil order is set, and ``design_explicit_fd`` sizes ``dt`` from it via
    ``rockem.utils.explicit_em_cfl_dt``.
    """
    order = int(p.fd_order) if p.fd_order is not None else resolve_fd_order_from_cfg(
        TEMPLATES_DIR / "mod.cfg", default=2
    )
    lpml = int(p.lpml) if p.lpml is not None else _template_cfg_int("lpml", 13)
    if lpml < order + 5:
        raise ValueError(
            f"lpml={lpml} violates the engine's lpml >= order + 5 requirement for order={order} "
            f"(needs >= {order + 5})."
        )

    min_off, max_off = _offset_bounds(p)
    out = design_explicit_fd(
        ExplicitDesignInputs(
            f_min_hz=p.resolved_f_min(),
            f_max_hz=p.resolved_f_max(),
            rho_min_ohm_m=float(p.rho_min_ohm_m),
            rho_max_ohm_m=float(p.rho_max_ohm_m),
            min_offset_m=min_off,
            max_offset_m=max_off,
            points_per_skin=int(p.points_per_skin),
            cells_per_min_offset=int(p.cells_per_min_offset),
            tan_delta_floor=float(p.tan_delta_floor),
            eps_r_cap=float(p.eps_r_cap),
            sigma_max_clip_s_per_m=p.sigma_max_clip_s_per_m,
            lpml=lpml,
            max_depth_offset_m=float(max_depth_offset_m),
            fd_order=order,
        )
    )
    apertx = (float(p.apertx_override_m) if p.apertx_override_m is not None
              else float(out.apertx_m))
    design = {
        "dx_m": float(out.dx_m),
        "dt_s": float(out.dt_s),
        "eps_r_used": float(out.eps_r_used),
        "explicit_cfl_safety": float(out.explicit_cfl_safety),
        "fd_order": int(out.fd_order),
        "eps_r_cap_binding": bool(out.eps_r_cap_binding),
        "apertx_m": apertx,
        "apertx_from_design_m": float(out.apertx_m),
        "apertx_overridden": p.apertx_override_m is not None,
        "lpml": int(out.lpml),
        "pml_kmax": float(out.pml_kmax),
        "pml_smax": float(out.pml_smax),
        "pml_amax": float(out.pml_amax),
        "min_offset_m": float(min_off),
        "max_offset_m": float(max_off),
        "delta_induction_m": float(out.delta_induction_m),
        "depth_margin_m": float(out.depth_margin_m),
        "min_domain_halfdepth_m": float(out.min_domain_halfdepth_m),
        "notes": out.notes,
    }
    return design, order, lpml


def build_forward_inputs(out_dir: Path | str, p: SetupParams, *, verbose: bool = True) -> dict:
    """Write a complete, runnable forward directory (Step 01 'Finalize setup').

    Produces ``sg.rss``, ``ep.rss``, ``wav2d.rss``, ``Survey.rss``, ``mod.cfg``
    and ``setup_metadata.json`` under ``out_dir``, exactly as notebook 01's
    ``on_apply_outputs`` does, and returns the setup metadata dict.
    """
    from scripts.modules.rockem_bridge import utils as rockem_utils  # noqa: F401 (env setup)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_intermediate"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model = read_resistivity_from_segy(str(p.segy_path))
    if p.rho_min_ohm_m is None or p.rho_max_ohm_m is None:
        positive = np.asarray(model["resistivity"], dtype=float)
        positive = positive[positive > 0.0]
        p = replace(
            p,
            rho_min_ohm_m=float(np.nanmin(positive)) if p.rho_min_ohm_m is None else p.rho_min_ohm_m,
            rho_max_ohm_m=float(np.nanmax(positive)) if p.rho_max_ohm_m is None else p.rho_max_ohm_m,
        )

    design, order, lpml = fd_design_for(p, max_depth_offset_m=abs(p.rz0_m - p.tz0_m))
    eps_r_used = design["eps_r_used"]
    target_dx = design["dx_m"]
    target_dt = design["dt_s"]

    # --- wavelet (intermediate at wavelet_dt_s, then sinc-resampled to dt) ---
    wav_raw = tmp_dir / "wav2d_raw.rss"
    wav_params = create_wavelet_rss(
        flist=[float(f) for f in p.flist_hz],
        dt=float(p.wavelet_dt_s),
        n_periods=int(p.n_periods),
        alpha=float(p.alpha),
        wavfile=str(wav_raw),
        dim=2,
        show_plot=False,
    )

    # --- model, padded in depth so the PML keeps its clearance ---
    padded, padded_oz, pad_info = pad_resistivity_for_depth_margin(
        model["resistivity"], model["oz"], model["dz"],
        source_z_m=float(p.tz0_m),
        min_domain_halfdepth_m=design["min_domain_halfdepth_m"],
    )
    if pad_info["padded"]:
        model["resistivity"] = padded
        model["oz"] = padded_oz
        model["z"] = padded_oz + model["dz"] * np.arange(padded.shape[0])

    sg_raw = tmp_dir / "sg_raw.rss"
    ep_raw = tmp_dir / "ep_raw.rss"
    write_sg_ep_rss(
        model["resistivity"], model["dx"], model["dz"], model["ox"], model["oz"],
        str(sg_raw), str(ep_raw), ep_value=eps_r_used, ny_samples=1,
    )

    # NEAREST, not linear. A linear resample turns every material interface into
    # a ONE-CELL RAMP whose effective depth is the ramp midpoint, which sits
    # between cell faces - so the FD model's interface is not at any depth the
    # 1D inversion's candidate model can be snapped to, and `C(f)` cannot absorb
    # the difference (C is one complex number per frequency shared by every
    # transmitter; the error is model- and depth-dependent). Nearest puts the
    # interface exactly on a cell boundary. Measured on a 1 m -> 0.8 m resample
    # of a 2/100 Ohm-m step: linear leaves one intermediate cell at 80.4 Ohm-m,
    # nearest leaves none. `interpolate_rss_python` forces antialias off here.
    #
    # This changes the forward model, so datasets and calibrations built before
    # it are NOT comparable with ones built after. Notebook 01 does the same two
    # resamples in its own `on_apply_outputs`; the two must not diverge.
    interpolate_rss_python(sg_raw, out_dir / "sg.rss", d1f=target_dx, d3f=target_dx, method="nearest")
    interpolate_rss_python(ep_raw, out_dir / "ep.rss", d1f=target_dx, d3f=target_dx, method="nearest")
    interpolate_rss_python(wav_raw, out_dir / "wav2d.rss", d1f=target_dt, method="sinc")
    enforce_rss_min_value(out_dir / "sg.rss", min_value=1e-8)
    enforce_rss_min_value(out_dir / "ep.rss", min_value=1e-8)

    # --- survey ---
    survey_cfg = tmp_dir / "survey.cfg"
    shutil.copyfile(TEMPLATES_DIR / "survey.cfg", survey_cfg)
    update_survey_cfg(survey_cfg, {
        "dim": 2, "sx0": p.tx0_m, "sy0": 0.0, "sz0": p.tz0_m,
        "dsx": p.dtx_m, "dsy": 1.0, "nsx": p.ntx, "nsy": 1,
        "gx0": p.rx0_m, "gy0": 0.0, "gz0": p.rz0_m,
        "dgx": p.drx_m, "dgy": 1.0, "ngx": p.nrx, "ngy": 1,
    })
    survey_meta = generate_survey_rss(survey_cfg.parent, cfg_filename="survey.cfg",
                                      output_filename="Survey.rss")
    shutil.copyfile(survey_meta["survey_rss"], out_dir / "Survey.rss")

    # --- mod.cfg ---
    mod_cfg = out_dir / "mod.cfg"
    shutil.copyfile(TEMPLATES_DIR / "mod.cfg", mod_cfg)
    source_field = str(p.source_field).upper()
    if source_field not in SOURCE_TYPE_CODES:
        raise ValueError(f"source_field must be one of {sorted(SOURCE_TYPE_CODES)}, got {source_field!r}")
    update_cfg_values(mod_cfg, {
        "order": str(int(order)),
        "lpml": str(int(lpml)),
        "source_type": str(SOURCE_TYPE_CODES[source_field]),
    })
    # dtrec follows the INTERMEDIATE wavelet sampling (1e-5 s by default), which
    # is 394x coarser than the model dt. That is a deliberate storage choice, but
    # it means the production shot record and the wavelet have DIFFERENT lengths
    # and different sample grids, while the calibration runs (which write
    # dtrec = dt_model) have identical ones. `fd_visualization.steady_state_gains`
    # therefore has to align the two analysis windows on ABSOLUTE time and
    # correct the sub-sample remainder explicitly - see the note there. Before
    # that was done, this mismatch put a frequency-proportional phase error of up
    # to 21.6 deg into every production channel gain, invisible to C(f) because
    # the calibration geometry has no offset at all.
    update_modcfg_for_workshop(
        mod_cfg,
        dtrec_s=float(p.wavelet_dt_s),
        apertx_m=design["apertx_m"],
        sg_path="sg.rss", ep_path="ep.rss",
        wavelet_path="wav2d.rss", survey_path="Survey.rss",
        pml_kmax=design["pml_kmax"], pml_smax=design["pml_smax"], pml_amax=design["pml_amax"],
    )
    for name in ("runmod.sh", "clean.sh"):
        dest = out_dir / name
        shutil.copyfile(TEMPLATES_DIR / name, dest)
        os.chmod(dest, 0o755)

    config = load_config()
    meta = {
        "flist_hz": [float(f) for f in p.flist_hz],
        "dt_wavelet_s": float(p.wavelet_dt_s),
        "ntx": int(p.ntx), "nrx": int(p.nrx),
        "tx0_m": float(p.tx0_m), "tz0_m": float(p.tz0_m), "dtx_m": float(p.dtx_m),
        "rx0_m": float(p.rx0_m), "rz0_m": float(p.rz0_m), "drx_m": float(p.drx_m),
        "dx_model_target_m": float(target_dx),
        "dt_model_target_s": float(target_dt),
        "dtrec_written_s": float(p.wavelet_dt_s),
        "forward_data_dim": 2,
        "forward_engine": config.forward_engine_te2d(),
        "forward_cfg": "mod.cfg",
        "forward_wavelet": "wav2d.rss",
        "ny_samples": 1,
        "f_min_hz": p.resolved_f_min(),
        "f_max_hz": p.resolved_f_max(),
        # NOT the wavelet's own n_periods: that asks for the whole record and
        # defeats steady_state_phasor's ramp-up skip (see
        # source.create_wavelet_rss's n_periods_extract_safe - measured 10x
        # worse frequency drift in |C|/dx^2).
        "n_periods_extract": float(wav_params["n_periods_extract_safe"]),
        "wavelet_n_periods": float(wav_params["n_periods"]),
        "wavelet_ramp_seconds": float(wav_params["ramp_seconds"]),
        "rho_min_ohm_m": float(p.rho_min_ohm_m),
        "rho_max_ohm_m": float(p.rho_max_ohm_m),
        "eps_r_used": float(eps_r_used),
        "explicit_cfl_safety": float(design["explicit_cfl_safety"]),
        "fd_order": int(order),
        "source_field": source_field,
        "source_type": int(SOURCE_TYPE_CODES[source_field]),
        "eps_r_cap_binding": bool(design["eps_r_cap_binding"]),
        "apertx_m": float(design["apertx_m"]),
        "apertx_from_design_m": float(design["apertx_from_design_m"]),
        "apertx_overridden": bool(design["apertx_overridden"]),
        "min_offset_m": float(design["min_offset_m"]),
        "max_offset_m": float(design["max_offset_m"]),
        "wavelet_nt": int(wav_params["nt"]),
        "wavelet_rec_time_s": float(wav_params["rec_time_actual"]),
        "nt_model": int(round(float(wav_params["rec_time_actual"]) / target_dt)) + 1,
        "pml_heuristic": {
            "source": "design_explicit_fd (explicit-engine, eps_r_used-based vmax)",
            "lpml_cells": int(lpml),
            "f0_hz": float(max(p.flist_hz)),
            "vmax_m_per_s": float(C0 / np.sqrt(eps_r_used)),
            "pml_kmax": design["pml_kmax"],
            "pml_smax": design["pml_smax"],
            "pml_amax": design["pml_amax"],
        },
        "segy_template_path": str(Path(p.segy_path).expanduser().resolve()),
        "segy_ox": float(model["ox"]), "segy_oz": float(model["oz"]),
        "segy_dx": float(model["dx"]), "segy_dz": float(model["dz"]),
        "segy_nx": int(np.asarray(model["x"]).size),
        "segy_nz": int(np.asarray(model["z"]).size),
        "fd_design_notes": design["notes"],
    }
    (out_dir / "setup_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    if verbose:
        from rockem.utils.utils import stencil_cfl_factor
        kappa = stencil_cfl_factor(order)
        print(
            f"[setup] {out_dir}  order={order} (kappa={kappa:.5f})  lpml={lpml}  src={source_field}"
            f"  band=[{p.resolved_f_min():g},{p.resolved_f_max():g}] Hz  tones={list(p.flist_hz)}\n"
            f"        dx={target_dx:.4f} m  dt={target_dt:.6e} s  eps_r={eps_r_used:.1f}"
            f"{' (CAP BINDING)' if design['eps_r_cap_binding'] else ''}"
            f"  rec_time={wav_params['rec_time_actual']:.4e} s  nt={meta['nt_model']:,}"
            f"  apertx={design['apertx_m']:.2f} m"
            + (f" (OVERRIDDEN from {design['apertx_from_design_m']:.2f} m; "
               f"half-width {0.5*design['apertx_m']:.1f} m)"
               if design["apertx_overridden"] else "")
        )
    return meta


def run_forward(run_dir: Path | str, *, cfg_name: str = "mod.cfg", nproc: int = 6,
                verbose: bool = True) -> dict:
    """Run the explicit TE2D engine in ``run_dir``; return timing + rc.

    Echoes the cfg's ``order``/``lpml``/``source_type`` read back from disk
    immediately before launching, so an A/B that was silently inert is visible
    in the log rather than only in a bit-identical result.
    """
    run_dir = Path(run_dir)
    config = load_config()
    engine = config.binary_path(config.forward_engine_te2d())
    if not engine.is_file():
        raise FileNotFoundError(f"Forward engine not found: {engine}")

    cfg = read_cfg_values(run_dir / cfg_name)
    if verbose:
        print(f"[run ] {run_dir.name}/{cfg_name}: order={cfg.get('order')} lpml={cfg.get('lpml')} "
              f"source_type={cfg.get('source_type')} apertx={cfg.get('apertx')} -np {nproc}")
    (run_dir / "Data").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    proc = subprocess.run(
        [config.mpirun, "-np", str(int(nproc)), str(engine), cfg_name],
        cwd=str(run_dir), capture_output=True, text=True,
    )
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Forward engine exited {proc.returncode} in {run_dir}:\n"
            + ((proc.stdout or "") + (proc.stderr or ""))[-2000:]
        )
    if verbose:
        print(f"[run ] done in {wall_s:.1f} s ({wall_s/60:.2f} min)")
    return {"wall_s": wall_s, "returncode": proc.returncode, "nproc": int(nproc),
            "cfg": {k: cfg.get(k) for k in ("order", "lpml", "source_type", "apertx")}}


def run_calibration(fwd_dir: Path | str, *, method: str, nproc: int = 6,
                    source_field: str = "HX",
                    freqs_hz: Optional[Sequence[float]] = None,
                    save_to_metadata: bool = True, verbose: bool = True) -> dict:
    """Prepare + run + fit one FDTD-vs-analytic calibration (notebook 02).

    ``source_field`` selects the magnetic source component: ``"HX"`` (Kx,
    ``source_type=3``) or ``"HZ"`` (Kz, ``source_type=5``). Each gets its own
    run directory and its own fitted ``C(f)``.
    """
    from scripts.modules.fdtd_analytic_calibration import (
        CALIBRATION_CFG_NAME,
        METHOD_HOMOGENEOUS,
        METHOD_LATERAL_AVERAGE,
        compute_calibration_from_fdtd_outputs,
        load_setup_metadata,
        prepare_homogeneous_calibration_run,
        prepare_lateral_average_calibration_run,
        save_calibration_to_metadata,
    )

    fwd_dir = Path(fwd_dir)
    meta_path = fwd_dir / "setup_metadata.json"
    meta = load_setup_metadata(meta_path)
    freqs = list(freqs_hz) if freqs_hz is not None else [float(f) for f in meta["flist_hz"]]

    if method == METHOD_LATERAL_AVERAGE:
        prep = prepare_lateral_average_calibration_run(fwd_dir, meta_path, source_field=source_field)
    elif method == METHOD_HOMOGENEOUS:
        prep = prepare_homogeneous_calibration_run(fwd_dir, meta_path, source_field=source_field)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    run_dir = Path(prep["run_dir"])
    if verbose:
        print(f"[cal ] {method} src={prep['source_field']}: domain "
              f"{prep['lx']:.1f} x {prep['lz']:.1f} m ({prep['nx']}x{prep['nz']} cells)")
    timing = run_forward(run_dir, cfg_name=CALIBRATION_CFG_NAME, nproc=nproc, verbose=verbose)

    cal = compute_calibration_from_fdtd_outputs(
        fwd_dir, meta_path, freqs,
        f_min_hz=float(meta["f_min_hz"]),
        n_periods_extract=float(meta["n_periods_extract"]),
        method=method, source_field=source_field,
    )
    cal["fdtd_wall_s"] = timing["wall_s"]
    if save_to_metadata:
        written = save_calibration_to_metadata(meta_path, cal)
        # Surfaced rather than swallowed: a batch loop is exactly where a
        # mixed-Earth-model file gets built without anyone noticing.
        warn = written.get("calibration_consistency_warning")
        if warn:
            cal["consistency_warning"] = warn
            if verbose:
                print(f"[cal ] WARNING: {warn}")
    if verbose:
        print(format_calibration_table(cal))
    return cal


def run_calibration_matrix(out_root: Path | str, *, method: str, nproc: int = 6,
                           datasets: Optional[Sequence[Mapping[str, Any]]] = None,
                           sources: Optional[Sequence[str]] = None,
                           save_to_metadata: bool = True,
                           verbose: bool = True,
                           on_progress=None) -> list[dict]:
    """Calibrate EVERY dataset of the acquisition matrix, sequentially.

    Choosing N frequencies and both sources in Step 01 must not turn into 2N
    rounds of picking a dataset, picking a source and pressing a button. Step 01
    already recorded what was selected; this walks that list.

    By default each dataset is calibrated with **its own** source - the one it
    was modelled with - which is what the 1D tensor inversion then reads back
    per source (`inversion_1d.tensor_calibration` takes one metadata path per
    source). Pass `sources` to calibrate every dataset with a fixed set instead.

    ONE `method` for all of them, deliberately: mixing methods across sources is
    the mistake `fdtd_analytic_calibration.calibration_consistency_warning`
    exists to catch, and a batch loop is the easiest possible way to commit it.

    `on_progress(i, n, dataset, cal_or_exception)` is called after each dataset,
    so a GUI can report progress without this function importing ipywidgets.

    A failure on one dataset does not abandon the rest: it is recorded in that
    entry's `"error"` and the loop continues, because a two-hour batch that
    stops on the first bad dataset is worse than one that tells you which failed.
    """
    out_root = Path(out_root)
    ds = list(datasets) if datasets is not None else iter_datasets(out_root)
    if not ds:
        raise FileNotFoundError(
            f"No forward datasets under {out_root}. Run Step 01, then Step 02 modelling."
        )

    results: list[dict] = []
    for i, d in enumerate(ds, start=1):
        wanted = ([str(s).upper() for s in sources] if sources is not None
                  else [str(d.get("source_field", "HX")).upper()])
        for src in wanted:
            label = f"{d['name']} [{src}]"
            if verbose:
                print(f"\n[{i}/{len(ds)}] calibrating {label} ({method})")
            entry = {"name": d["name"], "run_dir": str(d["run_dir"]),
                     "source_field": src, "method": method}
            try:
                cal = run_calibration(d["run_dir"], method=method, nproc=nproc,
                                      source_field=src,
                                      save_to_metadata=save_to_metadata,
                                      verbose=verbose)
                entry["cal"] = cal
                entry["ok"] = True
                if cal.get("consistency_warning"):
                    entry["warning"] = cal["consistency_warning"]
            except Exception as exc:                      # noqa: BLE001
                entry["ok"] = False
                entry["error"] = f"{type(exc).__name__}: {exc}"
                if verbose:
                    print(f"       FAILED: {entry['error']}")
            results.append(entry)
            if on_progress is not None:
                on_progress(len(results), len(ds) * len(wanted), d, entry)

    if verbose:
        n_ok = sum(1 for r in results if r.get("ok"))
        print(f"\nCalibrated {n_ok}/{len(results)} dataset-source pairs with {method}.")
        for r in results:
            if not r.get("ok"):
                print(f"  FAILED {r['name']} [{r['source_field']}]: {r['error']}")
        for r in results:
            if r.get("warning"):
                print(f"  WARNING {r['name']} [{r['source_field']}]: {r['warning']}")
    return results


def format_calibration_table(cal) -> str:
    """One-line-per-frequency |C|/dx^2 / phase / scatter table."""
    c = np.asarray(cal["C_hxhz_shared"], dtype=complex)
    dx2 = float(cal["dx_squared"])
    lines = [
        f"  method={cal.get('method')}  source={cal.get('source_field', 'HX')}  "
        f"dx={cal['dx_m']:.4f} m  dx^2={dx2:.6g}",
        "     f [Hz]    |C|/dx^2   phase [deg]   Hx scat [%]   Hz scat [%]",
    ]
    for i, f in enumerate(cal["freqs_hz"]):
        lines.append(
            f"  {f:9.0f}   {abs(c[i])/dx2:9.5f}   {np.angle(c[i], deg=True):+11.3f}   "
            f"{cal['scatter_hx_pct'][i]:11.4f}   {cal['scatter_hz_pct'][i]:11.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 2: per-frequency runs
# ---------------------------------------------------------------------------
#
# Two design rules pull in OPPOSITE directions:
#
#   * spatial sampling is set by the HIGHEST frequency - dx must resolve the
#     smallest skin depth, and delta ~ 1/sqrt(f), so high f forces a fine grid;
#   * record length is set by the LOWEST frequency - the extraction needs
#     n_periods_extract periods of the slowest tone, and T ~ 1/f, so low f
#     forces a long record.
#
# Modelling the whole band in ONE broadband run applies the fine grid of the
# highest tone through the long record of the lowest one. Neither tone needs
# that combination; it is an artefact of bundling them. Splitting the band into
# one independent run per frequency removes it - measured at ~2.1x fewer
# cell-steps in total on this workshop's own survey (see
# scripts/experiments/per_frequency_cost.py), and because the runs are
# completely independent they can also go concurrently.
#
# The saving is bounded by two things worth knowing before relying on it:
#   * the eps_r cap binds at the low end (eps_r ~ sigma_min/(tan_delta_floor *
#     omega_max * eps0) grows as f falls, and clips at eps_r_cap), which throws
#     away part of the low-frequency time-step gain;
#   * the MINIMUM-OFFSET rule binds dx at 1 kHz (dx = min(induction limit,
#     min_offset/cells_per_min_offset)) - a geometric requirement independent
#     of frequency, so the low-frequency saving is correctly capped by the
#     survey geometry. Do NOT relax cells_per_min_offset to chase it.
#
# NOT evaluated here, deliberately: replacing the broadband wavelet + Fourier
# extraction with a narrowband or single-frequency run to steady state. Once
# each run is monochromatic in purpose that may well be cheaper still, and it
# removes the n_periods_extract choice entirely - but it is a SECOND change,
# and bundling it with the split would make it impossible to attribute any
# difference in C(f) to one or the other. Evaluate it after the split is
# measured, not alongside it.


def build_per_frequency_forward_inputs(
    out_root: Path | str,
    p: SetupParams,
    *,
    verbose: bool = True,
) -> dict:
    """One independent, fully-designed forward run per tone in ``p.flist_hz``.

    Each frequency gets its own ``design_explicit_fd`` call with
    ``f_min = f_max = f``, hence its own dx, dt, eps_r and PML parameters, its
    own resampled ``sg.rss``/``ep.rss``, its own survey snapped to that grid,
    its own run directory (``<out_root>/f{f:.0f}Hz``) and its own
    ``setup_metadata.json``.

    Returns a manifest dict (also written to ``<out_root>/manifest.json``)
    with the per-frequency metadata and the total predicted cost, so the split
    can be A/B'd against the equivalent single broadband run without re-deriving
    anything.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    runs = {}
    for f in [float(v) for v in p.flist_hz]:
        run_dir = out_root / f"f{f:.0f}Hz"
        sub = replace(p, flist_hz=(f,), f_min_hz=f, f_max_hz=f)
        runs[f"{f:.0f}"] = {
            "run_dir": str(run_dir),
            "freq_hz": f,
            "meta": build_forward_inputs(run_dir, sub, verbose=verbose),
        }
    manifest = {
        "mode": "per_frequency",
        "flist_hz": [float(v) for v in p.flist_hz],
        "source_field": str(p.source_field).upper(),
        "runs": runs,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if verbose:
        total = sum(r["meta"]["nt_model"] for r in runs.values())
        print(f"[split] {len(runs)} independent runs under {out_root}; "
              f"total nt across runs = {total:,}")
    return manifest


def load_manifest(out_root: Path | str) -> dict:
    return json.loads((Path(out_root) / "manifest.json").read_text())


# ---------------------------------------------------------------------------
# The full acquisition matrix: N frequencies x M source components
# ---------------------------------------------------------------------------
#
# Task 2 splits the band into independent per-frequency runs; Task 5 adds Kz
# alongside Kx. Together they define the workshop's actual target acquisition:
# one FDTD run per (frequency, source component) pair. With the shipped
# defaults that is 4 x 2 = 8 runs, and it is what produces the complete 2x2
# magnetic coupling matrix at every frequency:
#
#     Kx source -> Hx, Hz     (Cxx, Cxz)
#     Kz source -> Hx, Hz     (Czx, Czz)
#
# Both receiver components are recorded by every run already (mod.cfg sets
# Hxrecord and Hzrecord true), so the ONLY thing that needs to vary between
# runs is source_type - and the frequency-dependent grid, if the band is split.
#
# THERE IS NO BROADBAND RUN. Every dataset is one frequency and one source:
# spatial sampling is set by the HIGHEST tone and record length by the LOWEST,
# so a broadband run applies the fine grid of the top tone through the long
# record of the bottom one and pays for both. Measured on this survey: 312.1 s
# split against 621.9 s broadband, i.e. 1.99x serially and up to 5.84x
# concurrently.
#
# Each dataset therefore gets its own dx/dt/eps_r, its own single-tone wavelet
# and its own C - and the whole matrix is always built, modelled, calibrated and
# inverted TOGETHER. Nothing downstream offers a per-dataset action, because
# then the operator has to carry the matrix in their head and can leave the
# workspace half-processed or mixed across Earth models.
#
# Reading a legacy single-dataset workspace still works: `iter_datasets` falls
# back to treating `out_root` itself as one dataset. That is a READER for data
# already on disk, not a way to create one.


def dataset_name(freq_hz: Optional[float], source_field: str) -> str:
    """Directory name for one (frequency, source) dataset.

    `freq_hz=None` only ever names a LEGACY broadband dataset already on disk;
    nothing builds one any more.
    """
    src = str(source_field).lower()
    return f"{'broadband' if freq_hz is None else f'f{freq_hz:.0f}Hz'}_{src}"


def build_forward_matrix(
    out_root: Path | str,
    p: SetupParams,
    *,
    source_fields: Sequence[str] = ("HX", "HZ"),
    verbose: bool = True,
) -> dict:
    """Build every (frequency, source) forward dataset and a manifest.

    ALWAYS one dataset per frequency per source - see the note above. The
    ``split_by_frequency`` switch is gone: a broadband run is strictly more
    expensive and produces a grid that suits neither end of the band, so it is
    not an option the workshop offers.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sources = [str(s).upper() for s in source_fields]
    for s in sources:
        if s not in SOURCE_TYPE_CODES:
            raise ValueError(f"source_field must be one of {sorted(SOURCE_TYPE_CODES)}, got {s!r}")
    if not sources:
        raise ValueError("At least one source component must be selected.")

    freqs = [float(f) for f in p.flist_hz]
    if not freqs:
        raise ValueError("flist_hz is empty - Step 01 must specify at least one tone.")

    runs: dict[str, dict] = {}
    for f in freqs:
        for src in sources:
            sub = replace(p, source_field=src, flist_hz=(f,), f_min_hz=f, f_max_hz=f)
            name = dataset_name(f, src)
            run_dir = out_root / name
            runs[name] = {
                "run_dir": str(run_dir),
                "freq_hz": f,
                "source_field": src,
                "meta": build_forward_inputs(run_dir, sub, verbose=verbose),
            }

    manifest = {
        "mode": "per_frequency_per_source",
        "source_fields": sources,
        "flist_hz": [float(v) for v in p.flist_hz],
        "n_datasets": len(runs),
        "runs": runs,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if verbose:
        total_nt = sum(r["meta"]["nt_model"] for r in runs.values())
        print(f"[matrix] {len(runs)} dataset(s) under {out_root} "
              f"({len(freqs)} frequency(ies) x {len(sources)} source(s)); "
              f"total nt across runs = {total_nt:,}")
    return manifest


def _sg_half_sample_origin(sg_path: Path, dx: float) -> float:
    """One point of the lattice a NEAREST-resampled model can put an interface on.

    `sg.rss` samples sit at `oz + k*dz`, and a nearest resample makes an
    interface a step BETWEEN two samples - so its effective depth is
    `oz + (k + 1/2)*dz`. This returns `oz + dz/2`; anything congruent to it
    modulo `dz` would do. Measured by
    `scripts/experiments/interface_snapping.py`, which reports every FD
    interface as 0.0000 m off this lattice.

    Falls back to `dx/2` when the model has not been written yet, so a workspace
    that Step 01 has not finished still resolves rather than raising here.
    """
    try:
        from scripts.modules.multiscale_2d import read_sg_grid
        z = np.asarray(read_sg_grid(sg_path)["z"], dtype=float)
        return float(z[0] + 0.5 * (z[1] - z[0])) if z.size > 1 else float(z[0])
    except Exception:                                   # noqa: BLE001
        return 0.5 * float(dx)


def matrix_setup(out_root: Path | str) -> dict:
    """Everything Step 01 decided, resolved across the WHOLE acquisition matrix.

    ONE implementation, used by Steps 05 and 06, because these values are
    per-dataset and collapsing them to a band-wide scalar is a bug that is very
    hard to see. Measured cost of doing it with `eps_r_used` alone: 0.91 sigma
    of systematic bias on Hx at 6 kHz - most of the assumed noise budget, on the
    best-resolved data in the survey.

    Returns:
        datasets     the `iter_datasets` list
        by_source    {source: [run_dir, ...]} ordered by frequency
        meta_paths   {source: [setup_metadata.json, ...]} in the same order
        freqs        every frequency in the matrix, ascending
        eps_r        `eps_r_used` per frequency, aligned with `freqs`
        f_min        `f_min_hz` per frequency, aligned with `freqs`
        n_periods_extract   per frequency, aligned with `freqs`
        dx           `dx_model_target_m` per frequency, aligned with `freqs`
        snap_origin_m  one point of the lattice of interface depths that
                     frequency's FD model can actually represent, `oz + dz/2`.
                     `sg.rss` is resampled with NEAREST, so an interface is a
                     step between two samples and its effective depth is their
                     midpoint. The 1D inversion snaps candidate interfaces onto
                     this - see `analytic_1d_forward.snap_interfaces_to_grid`.
        representative      a metadata path that EXISTS, for scalar defaults
        single       True when there is nothing to assemble (one dataset)
    """
    out_root = Path(out_root)
    ds = iter_datasets(out_root)
    if not ds:
        raise FileNotFoundError(
            f"No forward datasets under {out_root}. Run Step 01 (Generate FD inputs) first."
        )

    rows: dict[float, dict] = {}
    by_source: dict[str, list] = {}
    for d in ds:
        meta = d.get("meta") or json.loads(
            (Path(d["run_dir"]) / "setup_metadata.json").read_text())
        src = str(d.get("source_field", "HX")).upper()
        by_source.setdefault(src, []).append((float(meta["flist_hz"][0]), Path(d["run_dir"])))
        dx = float(meta["dx_model_target_m"])
        for f in meta["flist_hz"]:
            rows.setdefault(float(f), {
                "eps_r": float(meta["eps_r_used"]),
                "f_min": float(meta["f_min_hz"]),
                "n_periods_extract": float(meta["n_periods_extract"]),
                "dx": dx,
                "snap_origin_m": _sg_half_sample_origin(Path(d["run_dir"]) / "sg.rss", dx),
            })
    for src in by_source:
        by_source[src] = [p for _f, p in sorted(by_source[src], key=lambda t: t[0])]

    freqs = np.asarray(sorted(rows), dtype=float)
    rep = Path(ds[0]["run_dir"]) / "setup_metadata.json"
    return {
        "datasets": ds,
        "by_source": by_source,
        "meta_paths": {s: [p / "setup_metadata.json" for p in v] for s, v in by_source.items()},
        "freqs": freqs,
        "eps_r": np.asarray([rows[float(f)]["eps_r"] for f in freqs], dtype=float),
        "f_min": np.asarray([rows[float(f)]["f_min"] for f in freqs], dtype=float),
        "n_periods_extract": np.asarray(
            [rows[float(f)]["n_periods_extract"] for f in freqs], dtype=float),
        "dx": np.asarray([rows[float(f)]["dx"] for f in freqs], dtype=float),
        "snap_origin_m": np.asarray(
            [rows[float(f)]["snap_origin_m"] for f in freqs], dtype=float),
        "representative": rep,
        "single": len(ds) < 2,
    }


@dataclass(frozen=True)
class DatasetPaths:
    """Every file that belongs to ONE dataset, resolved from its run directory.

    THE PATH-BINDING CONTRACT, in one place. On an acquisition-matrix workspace
    the forward ROOT holds only `manifest.json` and one subdirectory per
    dataset, so none of these files exists at the root - they all belong to a
    dataset. Notebooks 02/03/04/05/06 each used to re-join them by hand and
    rebind them through a per-notebook `global` list inside `_select_dataset`.
    That list was hand-maintained, and forgetting to extend it is exactly how
    `SETUP_META = <forward root>/setup_metadata.json` shipped and killed Step
    05's lambda tuner in front of a user. A second instance (`SG_TRUE_PATH`)
    survived the fix and was only found when `chainsweep.py` was written; a
    third (`FDMODEL_DATA_DIR` in notebook 05) was never rebound at all.

    So there is one mechanism now: a notebook holds a single `DS` and reads
    `DS.setup_meta`, `DS.hx`, ... There is no per-notebook list to forget.

    `chainsweep.py` walks the attributes of objects like this one, not just
    bare `Path` globals - keep it that way or the invariant stops being tested.
    """

    dir: Path
    setup_meta: Path
    sg: Path
    ep: Path
    wav2d: Path
    survey: Path
    mod_cfg: Path
    runmod: Path
    clean_sh: Path
    mpiqueue_log: Path
    data_dir: Path
    hx: Path
    hz: Path
    processed_dir: Path
    amp_phase_npz: Path


def dataset_paths(run_dir: Path | str) -> DatasetPaths:
    """Resolve every per-dataset artifact under ``run_dir``.

    Pure path arithmetic - nothing is required to exist, because a notebook
    binds these before Step 01 or Step 02 has produced them. Calibration
    directories are NOT here: `fdtd_analytic_calibration.calibration_run_dir`
    takes a method as well as a directory, so it stays where it is and is
    called with `DS.dir`.
    """
    d = Path(run_dir)
    data = d / "Data"
    processed = data / "processed"
    return DatasetPaths(
        dir=d,
        setup_meta=d / "setup_metadata.json",
        sg=d / "sg.rss",
        ep=d / "ep.rss",
        wav2d=d / "wav2d.rss",
        survey=d / "Survey.rss",
        mod_cfg=d / "mod.cfg",
        runmod=d / "runmod.sh",
        clean_sh=d / "clean.sh",
        mpiqueue_log=d / "mpiqueue.log",
        data_dir=data,
        hx=data / "Hxshot.rss",
        hz=data / "Hzshot.rss",
        processed_dir=processed,
        amp_phase_npz=processed / "amp_phase_results.npz",
    )


def select_dataset(out_root: Path | str, name: str | None = None):
    """`(entry, DatasetPaths)` for one dataset of ``out_root``, by name.

    Falls back to the first dataset when ``name`` does not match, and to the
    root itself when there are no datasets at all - so a workspace that Step 01
    has not written yet still binds to something rather than raising at import.
    Every notebook's dataset selector goes through this.
    """
    try:
        ds = iter_datasets(out_root)
    except Exception:                                   # noqa: BLE001
        ds = []
    chosen = next((d for d in ds if d["name"] == name), ds[0]) if ds else None
    run_dir = Path(chosen["run_dir"]) if chosen else Path(out_root)
    return chosen, dataset_paths(run_dir)


# Keys every `iter_datasets` entry is guaranteed to carry. Consumers (notebooks
# 02/04/06, `workshop_report`) index these directly, so anything that produces a
# dataset list must fill all of them.
DATASET_KEYS = ("name", "run_dir", "freq_hz", "source_field", "meta")


def iter_datasets(out_root: Path | str) -> list[dict]:
    """Every dataset in ``out_root``, from its manifest.

    Every entry carries all of `DATASET_KEYS`, whichever manifest schema it came
    from. That normalisation matters: `build_per_frequency_forward_inputs` (the
    older per-frequency writer) records no per-run `source_field` at all - it
    keeps one at the top level - so passing its entries through unchanged handed
    consumers a dict missing a key they index, and the only manifest of that
    shape is the one this workspace actually has.

    Falls back to treating ``out_root`` itself as one dataset when no manifest
    exists, so a workspace produced before the matrix was introduced still
    reads correctly.
    """
    out_root = Path(out_root)
    mpath = out_root / "manifest.json"
    if mpath.exists():
        man = json.loads(mpath.read_text())
        default_src = str(man.get("source_field", "HX")).upper()
        out = []
        for name, run in man.get("runs", {}).items():
            entry = dict(run, name=name)
            meta = entry.get("meta") or {}
            entry["source_field"] = str(
                entry.get("source_field") or meta.get("source_field") or default_src
            ).upper()
            entry.setdefault("freq_hz", meta.get("f_min_hz"))
            entry["run_dir"] = str(entry.get("run_dir") or (out_root / name))
            entry["meta"] = meta
            out.append(entry)
        return out
    if (out_root / "setup_metadata.json").exists():
        meta = json.loads((out_root / "setup_metadata.json").read_text())
        return [{
            "name": dataset_name(None, meta.get("source_field", "HX")),
            "run_dir": str(out_root), "freq_hz": None,
            "source_field": str(meta.get("source_field", "HX")).upper(), "meta": meta,
        }]
    return []


__all__ = [
    "DatasetPaths",
    "dataset_paths",
    "select_dataset",
    "DATASET_KEYS",
    "SOURCE_TYPE_CODES",
    "build_forward_matrix",
    "build_per_frequency_forward_inputs",
    "dataset_name",
    "iter_datasets",
    "matrix_setup",
    "load_manifest",
    "SetupParams",
    "build_forward_inputs",
    "fd_design_for",
    "format_calibration_table",
    "run_calibration",
    "run_calibration_matrix",
    "run_forward",
]
