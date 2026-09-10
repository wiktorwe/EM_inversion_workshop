"""Build and write per-run 1D inversion reports (JSON summary + REPORT.md)."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy/complex/Path values into JSON-serializable ones."""
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {"real": obj.real.tolist(), "imag": obj.imag.tolist()}
        return obj.tolist()
    if isinstance(obj, (complex, np.complexfloating)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def per_frequency_list(value) -> list[float]:
    """ALWAYS a list, aligned with `freqs_hz`. Never a bare float.

    `eps_r_used`, `f_min_hz` and `n_periods_extract` are PER FREQUENCY (measured
    eps_r 1198.3 / 599.2 / 399.4 at 2/4/6 kHz on the workshop survey). Wrapping
    them in `float()` was a TypeError in the GUI on every matrix workspace.

    This returned a float for one value and a list for several - which made the
    EXPORTED TYPE DEPEND ON THE DATA. That is the same shape-dependent contract
    that caused the original breakage: a consumer written against a
    single-dataset report gets a float, works, and then fails on a matrix. A
    one-element list on a single-dataset workspace costs nothing - every reader
    (`render_1d_run_report_md`, `format_run_parameters_html`,
    `workshop_report.format_run_parameters_html`, notebook 06's
    `eps_r_for_freqs`) prints or broadcasts it unchanged.
    """
    return [float(v) for v in np.asarray(value, dtype=float).reshape(-1)]


def _abs_c_list(c_arr) -> list[float]:
    c = np.asarray(c_arr, dtype=complex).reshape(-1)
    return [float(v) for v in np.abs(c)]


def calibration_summary_block(cal: Optional[Mapping[str, Any]]) -> dict:
    """Compact calibration block for reports (from notebook 05 cfg or loaded meta)."""
    if not isinstance(cal, dict) or cal.get("C") is None:
        return {
            "method": "none",
            "present": False,
            "notes": "No calibration loaded for this run.",
        }
    c = np.asarray(cal.get("C"), dtype=complex)
    freqs = np.asarray(cal.get("freqs_hz", []), dtype=float)
    return {
        "method": str(cal.get("method", "homogeneous_rho_min")),
        "present": True,
        "freqs_hz": [float(v) for v in freqs],
        "C_abs": _abs_c_list(c),
        "C_mean_abs": float(np.mean(np.abs(c))) if c.size else None,
        "sigma_hx": [float(v) for v in np.asarray(cal.get("sigma_hx", []), dtype=float)],
        "sigma_hz": [float(v) for v in np.asarray(cal.get("sigma_hz", []), dtype=float)],
        "notes": str(cal.get("notes", "")),
    }


def build_1d_run_summary(
    *,
    run_dir: Path | str,
    cfg: Mapping[str, Any],
    tx_ids: Sequence[int],
    n_runs: int,
    seeds: Sequence[int],
    misfit: Sequence[float],
    misfit_mean: Sequence[float],
    misfit_std: Sequence[float],
    chi2: Sequence[float],
    freqs_hz: Sequence[float],
    n_periods_extract: float | Sequence[float],
    f_min_hz: float | Sequence[float],
    setup_meta_path: Path | str,
    calibration: Optional[Mapping[str, Any]] = None,
    forward_data_dim: Optional[int] = None,
    hx_path: Optional[str] = None,
    hz_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> dict:
    """Full machine-readable summary for Run{N}."""
    cal_block = calibration_summary_block(calibration)
    return {
        "run_dir": str(run_dir),
        "timestamp": timestamp or datetime.datetime.now().isoformat(),
        "n_tx": int(len(tx_ids)),
        "tx_ids": [int(v) for v in tx_ids],
        "n_runs_per_tx": int(n_runs),
        "seed_base": int(cfg.get("seed", seeds[0] if seeds else 0)),
        "seeds": [int(s) for s in seeds],
        "freqs_hz": [float(v) for v in freqs_hz],
        "f_min_hz": per_frequency_list(f_min_hz),
        "n_periods_extract": per_frequency_list(n_periods_extract),
        "forward_data_dim": int(forward_data_dim) if forward_data_dim is not None else None,
        "hx_path": hx_path,
        "hz_path": hz_path,
        "eps_r_used": per_frequency_list(cfg.get("eps_r", 7.0)),
        "background_rho": float(cfg.get("background_rho", 10.0)),
        "n_layers": int(cfg.get("n_layers", 0)),
        "z_start": float(cfg.get("z_start", 0.0)),
        "z_end": float(cfg.get("z_end", 0.0)),
        "rho_min": float(10.0 ** float(cfg["log10_rho_min"])) if "log10_rho_min" in cfg else None,
        "rho_max": float(10.0 ** float(cfg["log10_rho_max"])) if "log10_rho_max" in cfg else None,
        "thk_min": float(cfg.get("thk_min")) if cfg.get("thk_min") is not None else None,
        "thk_max": float(cfg.get("thk_max")) if cfg.get("thk_max") is not None else None,
        "optimizer": str(cfg.get("optimizer", "")),
        "maxiter": int(cfg["maxiter"]) if cfg.get("maxiter") is not None else None,
        "popsize": int(cfg["popsize"]) if cfg.get("popsize") is not None else None,
        "maxfun": int(cfg["maxfun"]) if cfg.get("maxfun") is not None else None,
        "block_max_iter": int(cfg["block_max_iter"]) if cfg.get("block_max_iter") is not None else None,
        "component_weights": {c: float(cfg.get(f"w_{c}", 1.0))
                              for c in ("Cxx", "Cxz", "Czx", "Czz")},
        "reg_lambda": float(cfg.get("reg_lambda", 0.0)),
        "misfit": [float(v) for v in misfit],
        "misfit_mean": [float(v) for v in misfit_mean],
        "misfit_std": [float(v) for v in misfit_std],
        "chi2": [float(v) for v in chi2],
        "calibration": cal_block,
        "setup_metadata_path": str(setup_meta_path),
    }


def render_1d_run_report_md(summary: Mapping[str, Any]) -> str:
    """Human-readable REPORT.md body from a summary dict."""
    freqs = summary.get("freqs_hz") or []
    cal = summary.get("calibration") or {}
    lines = [
        f"# 1D inversion run report",
        "",
        f"- **Run directory:** `{summary.get('run_dir', '')}`",
        f"- **Timestamp:** {summary.get('timestamp', '')}",
        f"- **Transmitters:** {summary.get('n_tx', 0)} — ids {summary.get('tx_ids', [])}",
        f"- **Ensemble:** {summary.get('n_runs_per_tx', 1)} run(s)/Tx, seed_base={summary.get('seed_base')}, seeds={summary.get('seeds', [])}",
        "",
        "## Data extraction",
        "",
        f"- **Frequencies (Hz):** {', '.join(f'{float(f):g}' for f in freqs) if freqs else '(none)'}",
        f"- **f_min (Hz):** {summary.get('f_min_hz')}",
        f"- **n_periods_extract:** {summary.get('n_periods_extract')}",
        f"- **Forward data dim:** {summary.get('forward_data_dim')}",
        f"- **Hx path:** `{summary.get('hx_path')}`",
        f"- **Hz path:** `{summary.get('hz_path')}`",
        "",
        "## Physics / model",
        "",
        f"- **eps_r_used:** {summary.get('eps_r_used')}",
        f"- **background_rho (Ohm-m):** {summary.get('background_rho')}",
        f"- **n_layers:** {summary.get('n_layers')}",
        f"- **Depth window:** z_start={summary.get('z_start')}, z_end={summary.get('z_end')}",
        f"- **Rho bounds (Ohm-m):** [{summary.get('rho_min')}, {summary.get('rho_max')}]",
        f"- **Thickness bounds (m):** [{summary.get('thk_min')}, {summary.get('thk_max')}]",
        "",
        "## Optimizer / misfit",
        "",
        f"- **Optimizer:** {summary.get('optimizer')}",
        f"- **maxiter / popsize / maxfun:** {summary.get('maxiter')} / {summary.get('popsize')} / {summary.get('maxfun')}",
        f"- **block_max_iter:** {summary.get('block_max_iter')}",
        "- **Component weights:** " + ", ".join(
            f"{c}={v}" for c, v in (summary.get("component_weights") or {}).items()),
        f"- **Tikhonov lambda:** {summary.get('reg_lambda')}",
        "",
        "## Calibration (global C from notebook 02)",
        "",
        "Active `C(f)` from the last successful Step 02 Calibrate (homogeneous or lateral-average).",
        "",
        f"- **Present:** {cal.get('present')}",
        f"- **Method:** {cal.get('method')}",
        f"- **|C| mean:** {cal.get('C_mean_abs')}",
        f"- **|C| per freq:** {cal.get('C_abs')}",
        f"- **sigma_hx:** {cal.get('sigma_hx')}",
        f"- **sigma_hz:** {cal.get('sigma_hz')}",
        f"- **Notes:** {cal.get('notes')}",
        "",
        "## Results summary",
        "",
        f"- **Best misfit per Tx:** {summary.get('misfit')}",
        f"- **Mean misfit per Tx:** {summary.get('misfit_mean')}",
        f"- **chi2 per Tx:** {summary.get('chi2')}",
        "",
        "## Provenance",
        "",
        f"- **setup_metadata.json:** `{summary.get('setup_metadata_path')}`",
        "",
    ]
    return "\n".join(lines)


def write_1d_run_reports(run_dir: Path | str, summary: Mapping[str, Any]) -> dict:
    """Write analytic_1d_inversion_summary.json and REPORT.md under run_dir."""
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = json_safe(dict(summary))
    json_path = out / "analytic_1d_inversion_summary.json"
    md_path = out / "REPORT.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(render_1d_run_report_md(payload))
    return {"summary_json": json_path, "report_md": md_path}


# Version of the OBSERVED-DATA convention a 1D run was fitted against. Bump it
# whenever a change makes new channel gains incomparable with old ones, and say
# why here - the number is only useful if it can be traced to a reason.
#
#   1  the original convention: `steady_state_gains` took "the last N samples"
#      of each record, and `n_periods_extract` included the source ramp.
#   2  (current) extraction windows are aligned on ABSOLUTE time with an exact
#      sub-sample phase correction, and `n_periods_extract` excludes the ramp.
#      Runs made under convention 1 were fitting phases wrong by up to 21.6
#      degrees, plus ~173 degrees of aliasing at 6 kHz, so their chi-squared is
#      not comparable with anything produced now.
#
# Runs written before this stamp existed carry no `data_convention` key at all;
# notebook 06 treats a missing key as convention 1 and says so on load, which is
# the only way an old run can be told apart from a new one after the fact.
DATA_CONVENTION = 2


def write_1d_run_metadata(
    run_dir: Path | str,
    cfg: Mapping[str, Any],
    n_runs: int,
    seeds: Sequence[int],
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write run_metadata.json with a JSON-safe config (includes freqs if present)."""
    meta = {
        "run_dir": str(run_dir),
        "timestamp": datetime.datetime.now().isoformat(),
        "data_convention": int(DATA_CONVENTION),
        "n_runs_per_tx": int(n_runs),
        "seed_base": int(cfg.get("seed", 0)),
        "seeds": [int(s) for s in seeds],
        "config": json_safe(cfg),
    }
    if extra:
        meta.update(json_safe(dict(extra)))
    p = Path(run_dir) / "run_metadata.json"
    p.write_text(json.dumps(meta, indent=2) + "\n")
    return p


def format_run_parameters_html(summary: Mapping[str, Any]) -> str:
    """Compact HTML panel for notebook 06 after loading a run."""
    freqs = summary.get("freqs_hz") or []
    cal = summary.get("calibration") or {}
    freq_txt = ", ".join(f"{float(f):g}" for f in freqs) if freqs else "(unknown)"
    cal_note = cal.get("notes") or cal.get("method") or ""
    rows = [
        ("Frequencies (Hz)", freq_txt),
        ("n_periods_extract", summary.get("n_periods_extract")),
        ("Optimizer", summary.get("optimizer")),
        ("n_layers", summary.get("n_layers")),
        ("background_rho", summary.get("background_rho")),
        ("eps_r_used", summary.get("eps_r_used")),
        ("Component weights",
         " / ".join(f"{c}={v}" for c, v
                    in (summary.get("component_weights") or {}).items())),
        ("Tikhonov λ", summary.get("reg_lambda")),
        ("Rho bounds", f"[{summary.get('rho_min')}, {summary.get('rho_max')}]"),
        ("Depth window", f"[{summary.get('z_start')}, {summary.get('z_end')}]"),
        ("Calibration |C| mean", cal.get("C_mean_abs")),
        ("Calibration notes", cal_note),
        ("Timestamp", summary.get("timestamp")),
    ]
    body = "".join(
        f"<tr><td><b>{k}</b></td><td><code>{v}</code></td></tr>" for k, v in rows if v is not None
    )
    return (
        "<div style='margin:8px 0;padding:8px;border:1px solid #ccc;'>"
        "<b>Run parameters</b> (from this run's report — you do not need to re-enter them)"
        f"<table>{body}</table></div>"
    )


__all__ = [
    "build_1d_run_summary",
    "per_frequency_list",
    "calibration_summary_block",
    "format_run_parameters_html",
    "json_safe",
    "render_1d_run_report_md",
    "DATA_CONVENTION",
    "write_1d_run_metadata",
    "write_1d_run_reports",
]
