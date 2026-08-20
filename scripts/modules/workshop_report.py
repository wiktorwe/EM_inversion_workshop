"""Build a LaTeX workflow report from existing workshop workspace artifacts."""

from __future__ import annotations

import datetime
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from scripts.modules.fd import read_cfg_values
from scripts.modules.fd_visualization import (
    compute_gains_for_fd_outputs,
    load_rss_traces,
)
from scripts.modules.report_figures import (
    save_1d_chi2_figure,
    save_1d_obs_pred_figure,
    save_1d_obs_pred_vs_tx_figure,
    save_1d_rho_vs_depth_figure,
    save_1d_section_figure,
    save_2d_model_compare_figure,
    save_2d_slices_figure,
    save_amp_phase_vs_rx_figure,
    save_amp_phase_vs_tx_figure,
    save_calibration_c_figure,
    save_obs_vs_syn_figure,
    save_obs_vs_syn_vs_tx_figure,
    save_resistivity_survey_figure,
    save_wavelet_figure,
    survey_positions_from_meta,
)
from scripts.modules.rss_model import resistivity_from_sg_rss
from scripts.modules.setup_defaults import load_setup_metadata, setup_metadata_path
from scripts.modules.workshop_config import WorkshopConfig, load_config

_TABSPEC = (
    r">{\raggedright\arraybackslash}p{0.32\textwidth}"
    r">{\raggedright\arraybackslash}p{0.60\textwidth}"
)
RUN_DIR_PATTERN = re.compile(r"^Run(\d+)$")


def _display_path(path: Any, root: Path | None = None) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if root is not None:
        try:
            return str(p.resolve().relative_to(Path(root).resolve()))
        except Exception:
            pass
    text = str(p)
    if len(text) > 84:
        return text[:40] + "..." + text[-40:]
    return text
SG_UP_RE = re.compile(r"sg_up\.rss-(\d+)$")
HX_MOD_RE = re.compile(r"data_Hx_mod\.rss-(\d+)$")
HX_ALT_RE = re.compile(r"data_mod_HX\.rss-(\d+)$")

OPTMETHOD_LABEL = {
    "1": "L-BFGS",
    "2": "CG_FR",
    "3": "steepest descent",
    "4": "CG_PR",
}
MISFIT_LABEL = {
    "0": "difference",
    "1": "correlation",
    "2": "amplitude ratio",
    "3": "phase difference",
}
LINESEARCH_LABEL = {
    "1": "decrease",
    "2": "Armijo",
    "3": "Wolfe",
    "4": "strong Wolfe",
    "5": "MoreThuente",
}

_LATEX_ESCAPE = (
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
)


@dataclass
class ReportContext:
    root: Path
    cfg: WorkshopConfig
    fwd_dir: Path
    setup_meta: dict
    setup_meta_path: Path
    report_dir: Path
    figures_dir: Path
    run_2d: Optional[Path] = None
    run_1d: Optional[Path] = None
    notes: list[str] = field(default_factory=list)
    figures: dict[str, Path] = field(default_factory=dict)
    timestamp: str = ""


_UNICODE_ASCII = (
    ("≈", "~="),
    ("≃", "~="),
    ("–", "-"),
    ("—", "-"),
    ("−", "-"),
    ("×", "x"),
    ("°", " deg"),
    ("±", "+/-"),
    ("≤", "<="),
    ("≥", ">="),
    ("λ", "lambda"),
    ("χ", "chi"),
    ("μ", "u"),
    ("Ω", "Ohm"),
    ("²", "^2"),
    ("…", "..."),
    ("’", "'"),
    ("‘", "'"),
    ("“", "'"),
    ("”", "'"),
)


def latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    for src, dst in _UNICODE_ASCII:
        text = text.replace(src, dst)
    text = text.encode("ascii", "replace").decode("ascii")
    for src, dst in _LATEX_ESCAPE:
        text = text.replace(src, dst)
    # p{} table cells do not break at "/" or "\_" otherwise, so long paths
    # overflow the page (Overfull \hbox).
    text = text.replace("/", r"/\allowbreak{}")
    text = text.replace(r"\_", r"\_\allowbreak{}")
    return text


def fmt_value(value: Any) -> str:
    if value is None:
        return "---"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return "---"
        return ", ".join(fmt_value(v) for v in value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "---"
        if value.ndim == 1 and value.size <= 16:
            return ", ".join(fmt_value(v) for v in value.tolist())
        return f"array{tuple(int(n) for n in value.shape)}"
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if not np.isfinite(v):
            return "---"
        av = abs(v)
        if v == 0.0:
            return "0"
        if av < 1e-3 or av >= 1e5:
            return f"{v:.4e}"
        return f"{v:g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def list_run_dirs(root_dir: Path | str) -> list[tuple[int, Path]]:
    root = Path(root_dir)
    if not root.exists():
        return []
    out: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = RUN_DIR_PATTERN.match(child.name)
        if m:
            out.append((int(m.group(1)), child))
    out.sort(key=lambda item: item[0])
    return out


def resolve_run_dir(
    runs_dir: Path,
    spec: Optional[str],
    *,
    kind: str,
    required: bool,
) -> Optional[Path]:
    runs = list_run_dirs(runs_dir)
    if spec:
        raw = Path(spec).expanduser()
        if raw.is_dir():
            return raw.resolve()
        name = spec if str(spec).startswith("Run") else f"Run{spec}"
        cand = runs_dir / name
        if cand.is_dir():
            return cand
        raise FileNotFoundError(f"No {kind} inversion run at {cand}")
    if not runs:
        if required:
            raise FileNotFoundError(f"No {kind} inversion runs under {runs_dir}")
        return None
    return runs[-1][1]


def latest_sg_up_file(run_dir: Path) -> Optional[Path]:
    candidates = list(run_dir.glob("Results/sg_up.rss-*")) + list(run_dir.glob("sg_up.rss-*"))
    if not candidates:
        return None

    def _suffix(path: Path) -> int:
        m = SG_UP_RE.search(path.name)
        return int(m.group(1)) if m else -1

    return sorted(candidates, key=lambda p: (_suffix(p), p.name))[-1]


def available_synthetic_pairs(run_dir: Path) -> list[tuple[int, Path, Path]]:
    run_dir = Path(run_dir)
    pairs: list[tuple[int, Path, Path]] = []
    seen: set[tuple[int, str, str]] = set()

    def _add(idx: int, hx: Path, hz: Path) -> None:
        key = (int(idx), str(hx), str(hz))
        if key in seen or not hx.exists() or not hz.exists():
            return
        seen.add(key)
        pairs.append((int(idx), Path(hx), Path(hz)))

    for hx in run_dir.glob("data_Hx_mod.rss-*"):
        m = HX_MOD_RE.search(hx.name)
        if m:
            idx = int(m.group(1))
            _add(idx, hx, run_dir / f"data_Hz_mod.rss-{idx}")
    for hx in run_dir.glob("data_mod_HX.rss-*"):
        m = HX_ALT_RE.search(hx.name)
        if m:
            idx = int(m.group(1))
            _add(idx, hx, run_dir / f"data_mod_HZ.rss-{idx}")
    _add(-2, run_dir / "data_Hx_mod.rss", run_dir / "data_Hz_mod.rss")
    _add(-1, run_dir / "data_mod_HX.rss", run_dir / "data_mod_HZ.rss")
    pairs.sort(key=lambda item: item[0])
    return pairs


def _kv_table(rows: Sequence[tuple[str, Any]], caption: Optional[str] = None) -> str:
    body = "\n".join(
        f"{latex_escape(k)} & {latex_escape(fmt_value(v))} \\\\" for k, v in rows if v is not None
    )
    env = "longtable" if len(rows) > 18 else "tabular"
    bits = []
    if env == "longtable":
        bits.append(rf"\begin{{longtable}}{{{_TABSPEC}}}")
        bits.append(r"\toprule")
        bits.append(r"Parameter & Value \\")
        bits.append(r"\midrule")
        bits.append(r"\endfirsthead")
        bits.append(r"\toprule")
        bits.append(r"Parameter & Value \\")
        bits.append(r"\midrule")
        bits.append(r"\endhead")
        bits.append(body)
        bits.append(r"\bottomrule")
        if caption:
            bits.append(rf"\caption{{{latex_escape(caption)}}}")
        bits.append(r"\end{longtable}")
    else:
        bits.append(r"\begin{center}")
        bits.append(rf"\begin{{tabular}}{{{_TABSPEC}}}")
        bits.append(r"\toprule")
        bits.append(r"Parameter & Value \\")
        bits.append(r"\midrule")
        bits.append(body)
        bits.append(r"\bottomrule")
        bits.append(r"\end{tabular}")
        bits.append(r"\end{center}")
    return "\n".join(bits)


def _figure_block(rel_name: str, caption: str) -> str:
    return "\n".join(
        [
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width=0.95\textwidth,height=0.82\textheight,keepaspectratio]{{{rel_name}}}",
            rf"\caption{{{latex_escape(caption)}}}",
            r"\end{figure}",
        ]
    )


def _note_block(text: str) -> str:
    return rf"\textit{{{latex_escape(text)}}}" + "\n"


def _cfg_label(raw: Optional[str], mapping: Mapping[str, str]) -> Optional[str]:
    if raw is None:
        return None
    key = str(raw).strip().strip('"')
    label = mapping.get(key)
    return f"{key} ({label})" if label else key


def gains_from_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    freqs = np.asarray(data["freqs"], dtype=float)
    return {
        "freqs": freqs,
        "geometry": {
            "tx_idx_per_trace": np.asarray(data["tx_idx_per_trace"], dtype=int),
            "rx_idx_per_trace": np.asarray(data["rx_idx_per_trace"], dtype=int),
            "rx_local_idx_per_trace": np.asarray(data["rx_local_idx_per_trace"], dtype=int),
            "src_x": np.asarray(data["src_x"], dtype=float),
            "src_z": np.asarray(data["src_z"], dtype=float),
            "rx_x": np.asarray(data["rx_x"], dtype=float),
            "rx_z": np.asarray(data["rx_z"], dtype=float),
        },
        "Hx": {
            "freqs": freqs,
            "amp_mean": np.asarray(data["Hx_amp_mean"], dtype=float),
            "phi_mean_rad": np.asarray(data["Hx_phi_mean_rad"], dtype=float),
        },
        "Hz": {
            "freqs": freqs,
            "amp_mean": np.asarray(data["Hz_amp_mean"], dtype=float),
            "phi_mean_rad": np.asarray(data["Hz_phi_mean_rad"], dtype=float),
        },
    }


def load_or_compute_gains(ctx: ReportContext) -> Optional[dict]:
    npz = ctx.fwd_dir / "processed" / "amp_phase_results.npz"
    if npz.exists():
        return gains_from_npz(npz)
    meta = ctx.setup_meta
    wav_name = str(meta.get("forward_wavelet") or "wav2d.rss")
    hx = ctx.fwd_dir / "Data" / "Hxshot.rss"
    hz = ctx.fwd_dir / "Data" / "Hzshot.rss"
    wav = ctx.fwd_dir / wav_name
    if not (hx.exists() and hz.exists() and wav.exists()):
        return None
    freqs = meta.get("flist_hz") or meta.get("freqs_hz") or []
    if not freqs:
        ctx.notes.append("Modelled-data frequencies missing from setup_metadata.json.")
        return None
    f_min = float(meta.get("f_min_hz") or min(float(v) for v in freqs))
    n_periods = float(meta.get("n_periods_extract") or 3.0)
    return compute_gains_for_fd_outputs(
        hx, hz, wav, freqs=freqs, f_min_hz=f_min, n_periods_extract=n_periods
    )


def _complex_c_from_meta(meta: Mapping) -> Optional[np.ndarray]:
    cal = meta.get("fdtd_analytic_calibration")
    if not isinstance(cal, dict):
        return None
    if "C_hxhz_shared_real" in cal and "C_hxhz_shared_imag" in cal:
        re = np.asarray(cal["C_hxhz_shared_real"], dtype=float)
        im = np.asarray(cal["C_hxhz_shared_imag"], dtype=float)
        return re + 1j * im
    if "C_hxhz_shared" in cal:
        return np.asarray(cal["C_hxhz_shared"], dtype=complex)
    mag = cal.get("C_magnitude")
    phase = cal.get("C_phase_deg")
    if mag is not None and phase is not None:
        return np.asarray(mag, dtype=float) * np.exp(1j * np.deg2rad(np.asarray(phase, dtype=float)))
    return None


def _try_figure(ctx: ReportContext, key: str, fn, *args, **kwargs) -> Optional[Path]:
    try:
        path = fn(*args, **kwargs)
    except Exception as exc:
        ctx.notes.append(f"Figure '{key}' skipped: {exc}")
        return None
    ctx.figures[key] = path
    return path


def collect_fw_rows(ctx: ReportContext) -> list[tuple[str, list[tuple[str, Any]]]]:
    meta = ctx.setup_meta
    mod_cfg: dict[str, str] = {}
    survey_cfg: dict[str, str] = {}
    mod_path = ctx.fwd_dir / str(meta.get("forward_cfg") or "mod.cfg")
    survey_path = ctx.fwd_dir / "survey.cfg"
    if mod_path.exists():
        try:
            mod_cfg = read_cfg_values(mod_path)
        except Exception as exc:
            ctx.notes.append(f"Could not read {mod_path.name}: {exc}")
    if survey_path.exists():
        try:
            survey_cfg = read_cfg_values(survey_path)
        except Exception as exc:
            ctx.notes.append(f"Could not read survey.cfg: {exc}")
    pml = meta.get("pml_heuristic") or {}
    groups = [
        (
            "Frequencies and wavelet",
            [
                ("Frequencies (Hz)", meta.get("flist_hz")),
                ("f_min (Hz)", meta.get("f_min_hz")),
                ("f_max (Hz)", meta.get("f_max_hz")),
                ("n_periods_extract", meta.get("n_periods_extract")),
                ("Wavelet dt (s)", meta.get("dt_wavelet_s")),
                ("Wavelet file", meta.get("forward_wavelet")),
            ],
        ),
        (
            "Survey geometry",
            [
                ("ntx", meta.get("ntx") or survey_cfg.get("nsx")),
                ("nrx", meta.get("nrx") or survey_cfg.get("ngx")),
                ("tx0 (m)", meta.get("tx0_m") or survey_cfg.get("sx0")),
                ("tz0 (m)", meta.get("tz0_m") or survey_cfg.get("sz0")),
                ("dtx (m)", meta.get("dtx_m") or survey_cfg.get("dsx")),
                ("rx0 (m)", meta.get("rx0_m") or survey_cfg.get("gx0")),
                ("rz0 (m)", meta.get("rz0_m") or survey_cfg.get("gz0")),
                ("drx (m)", meta.get("drx_m") or survey_cfg.get("dgx")),
                ("min offset (m)", meta.get("min_offset_m")),
                ("max offset (m)", meta.get("max_offset_m")),
            ],
        ),
        (
            "FD design",
            [
                ("dx target (m)", meta.get("dx_model_target_m")),
                ("dt target (s)", meta.get("dt_model_target_s")),
                ("dtrec (s)", meta.get("dtrec_written_s") or mod_cfg.get("dtrec")),
                ("eps_r used", meta.get("eps_r_used")),
                ("explicit CFL safety", meta.get("explicit_cfl_safety")),
                ("FD order", meta.get("fd_order") or mod_cfg.get("order")),
                ("eps_r cap binding", meta.get("eps_r_cap_binding")),
                ("apertx (m)", meta.get("apertx_m") or mod_cfg.get("apertx")),
                ("lpml (cells)", pml.get("lpml_cells") or mod_cfg.get("lpml")),
                ("pml_kmax", pml.get("pml_kmax") or mod_cfg.get("pml_kmax")),
                ("pml_smax", pml.get("pml_smax") or mod_cfg.get("pml_smax")),
                ("pml_amax", pml.get("pml_amax") or mod_cfg.get("pml_amax")),
                ("rho_min (Ohm-m)", meta.get("rho_min_ohm_m")),
                ("rho_max (Ohm-m)", meta.get("rho_max_ohm_m")),
            ],
        ),
        (
            "Engine",
            [
                ("Forward engine", meta.get("forward_engine")),
                ("Forward cfg", meta.get("forward_cfg")),
                ("Data dimension", meta.get("forward_data_dim")),
                ("ny samples", meta.get("ny_samples")),
                ("SEG-Y template", _display_path(meta.get("segy_template_path"), ctx.root)),
            ],
        ),
    ]
    return groups


def collect_calibration_rows(meta: Mapping) -> tuple[list[tuple[str, Any]], Optional[dict]]:
    cal = meta.get("fdtd_analytic_calibration")
    if not isinstance(cal, dict):
        return [], None
    c_arr = _complex_c_from_meta(meta)
    rows = [
        ("Method", cal.get("method")),
        ("Notes", cal.get("notes")),
        ("rho_ref (Ohm-m)", cal.get("rho_ohm_m")),
        ("Frequencies (Hz)", cal.get("freqs_hz")),
        ("|C|", np.abs(c_arr) if c_arr is not None else cal.get("C_magnitude")),
        ("arg C (deg)", np.angle(c_arr, deg=True) if c_arr is not None else cal.get("C_phase_deg")),
        ("sigma_hx", cal.get("sigma_hx")),
        ("sigma_hz", cal.get("sigma_hz")),
        ("Scatter Hx (%)", cal.get("scatter_hx_pct")),
        ("Scatter Hz (%)", cal.get("scatter_hz_pct")),
        ("Computed at", cal.get("computed_at")),
    ]
    return rows, cal


def collect_2d_inv_rows(ctx: ReportContext) -> list[tuple[str, Any]]:
    run_dir = ctx.run_2d
    assert run_dir is not None
    inv_cfg: dict[str, str] = {}
    cfg_path = run_dir / "inv.cfg"
    if not cfg_path.exists():
        cfg_path = ctx.cfg.inv_2d_input_dir / "inv.cfg"
    if cfg_path.exists():
        try:
            inv_cfg = read_cfg_values(cfg_path)
        except Exception as exc:
            ctx.notes.append(f"Could not read inv.cfg: {exc}")
    setup_path = ctx.cfg.inv_2d_input_dir / "inversion_setup_metadata.json"
    setup: dict[str, Any] = {}
    if setup_path.exists():
        try:
            setup = json.loads(setup_path.read_text())
        except Exception as exc:
            ctx.notes.append(f"Could not read inversion_setup_metadata.json: {exc}")
    sg_up = latest_sg_up_file(run_dir)
    return [
        ("Run directory", run_dir.name),
        ("Initial model mode", setup.get("initial_model_mode")),
        ("max_iterations", inv_cfg.get("max_iterations") or setup.get("max_iterations")),
        ("apertx (m)", inv_cfg.get("apertx") or setup.get("apertx")),
        ("dtx (m)", inv_cfg.get("dtx") or setup.get("dtx")),
        ("dtz (m)", inv_cfg.get("dtz") or setup.get("dtz")),
        ("tik_sgregalpha", inv_cfg.get("tik_sgregalpha") or setup.get("tik_sgregalpha")),
        ("constrain", inv_cfg.get("constrain") or setup.get("constrain")),
        ("sg_min (S/m)", setup.get("sg_min")),
        ("sg_max (S/m)", setup.get("sg_max")),
        ("optmethod", _cfg_label(inv_cfg.get("optmethod"), OPTMETHOD_LABEL)),
        ("linesearch", _cfg_label(inv_cfg.get("linesearch"), LINESEARCH_LABEL)),
        ("misfit_type", _cfg_label(inv_cfg.get("misfit_type"), MISFIT_LABEL)),
        ("paramtype", inv_cfg.get("paramtype")),
        ("order", inv_cfg.get("order")),
        ("lpml", inv_cfg.get("lpml")),
        ("update_sg", inv_cfg.get("update_sg")),
        ("update_ep", inv_cfg.get("update_ep")),
        ("Latest sg_up", sg_up.name if sg_up else None),
    ]


def collect_1d_inv_rows(summary: Mapping, run_meta: Mapping) -> list[tuple[str, Any]]:
    cal = summary.get("calibration") or {}
    cfg = (run_meta or {}).get("config") or {}
    return [
        ("Run directory", Path(str(summary.get("run_dir", ""))).name or summary.get("run_dir")),
        ("Timestamp", summary.get("timestamp")),
        ("Transmitters", summary.get("n_tx")),
        ("Tx ids", summary.get("tx_ids")),
        ("Runs per Tx", summary.get("n_runs_per_tx") or run_meta.get("n_runs_per_tx")),
        ("Seeds", summary.get("seeds") or run_meta.get("seeds")),
        ("Frequencies (Hz)", summary.get("freqs_hz") or cfg.get("freqs_hz")),
        ("f_min (Hz)", summary.get("f_min_hz")),
        ("n_periods_extract", summary.get("n_periods_extract")),
        ("eps_r_used", summary.get("eps_r_used") or cfg.get("eps_r")),
        ("background_rho (Ohm-m)", summary.get("background_rho") or cfg.get("background_rho")),
        ("n_layers", summary.get("n_layers") or cfg.get("n_layers")),
        ("z_start (m)", summary.get("z_start") or cfg.get("z_start")),
        ("z_end (m)", summary.get("z_end") or cfg.get("z_end")),
        ("rho_min (Ohm-m)", summary.get("rho_min")),
        ("rho_max (Ohm-m)", summary.get("rho_max")),
        ("thk_min (m)", summary.get("thk_min") or cfg.get("thk_min")),
        ("thk_max (m)", summary.get("thk_max") or cfg.get("thk_max")),
        ("Optimizer", summary.get("optimizer") or cfg.get("optimizer")),
        ("maxiter / popsize / maxfun", f"{summary.get('maxiter')} / {summary.get('popsize')} / {summary.get('maxfun')}"),
        ("block_max_iter", summary.get("block_max_iter") or cfg.get("block_max_iter")),
        ("w_hxh", summary.get("w_hxh") or cfg.get("w_hxh")),
        ("w_hxhz", summary.get("w_hxhz") or cfg.get("w_hxhz")),
        ("Tikhonov lambda", summary.get("reg_lambda") or cfg.get("reg_lambda")),
        ("Calibration present", cal.get("present")),
        ("Calibration method", cal.get("method")),
        ("|C| mean", cal.get("C_mean_abs")),
    ]


def write_fw_figures(ctx: ReportContext) -> None:
    sg = ctx.fwd_dir / "sg.rss"
    if sg.exists():
        tx_x, tx_z, rx_x, rx_z = survey_positions_from_meta(ctx.setup_meta)
        survey_rss = ctx.fwd_dir / "Survey.rss"
        if survey_rss.exists():
            try:
                traces = load_rss_traces(survey_rss)
                src = np.column_stack((traces["src_x"], traces["src_z"]))
                rec = np.column_stack((traces["rx_x"], traces["rx_z"]))
                src_u = np.unique(np.round(src, 6), axis=0)
                rec_u = np.unique(np.round(rec, 6), axis=0)
                tx_x, tx_z = src_u[:, 0], src_u[:, 1]
                rx_x, rx_z = rec_u[:, 0], rec_u[:, 1]
            except Exception:
                pass
        x, z, rho = resistivity_from_sg_rss(sg)
        _try_figure(
            ctx,
            "fw_resistivity",
            save_resistivity_survey_figure,
            x,
            z,
            rho,
            ctx.figures_dir / "fw_resistivity.pdf",
            tx_x=tx_x,
            tx_z=tx_z,
            rx_x=rx_x,
            rx_z=rx_z,
        )
    else:
        ctx.notes.append("Forward resistivity figure skipped: sg.rss not found.")

    wav_name = str(ctx.setup_meta.get("forward_wavelet") or "wav2d.rss")
    wav_path = ctx.fwd_dir / wav_name
    if wav_path.exists():
        wav = load_rss_traces(wav_path)
        w = np.asarray(wav["data"], dtype=float)[:, 0]
        t = np.arange(w.size, dtype=float) * float(wav["dt"])
        _try_figure(
            ctx,
            "fw_wavelet",
            save_wavelet_figure,
            t,
            w,
            ctx.figures_dir / "fw_wavelet.pdf",
            flist_hz=ctx.setup_meta.get("flist_hz") or ctx.setup_meta.get("freqs_hz"),
        )
    else:
        ctx.notes.append(f"Wavelet figure skipped: {wav_name} not found.")


def write_modelled_data_figures(ctx: ReportContext) -> None:
    gains = load_or_compute_gains(ctx)
    if gains is None:
        ctx.notes.append(
            "Modelled-data figures skipped: no processed/amp_phase_results.npz and no Hx/Hz shot gathers."
        )
        return
    _try_figure(
        ctx,
        "fw_amp_phase",
        save_amp_phase_vs_rx_figure,
        gains,
        ctx.figures_dir / "fw_amp_phase.pdf",
    )
    _try_figure(
        ctx,
        "fw_amp_phase_vs_tx",
        save_amp_phase_vs_tx_figure,
        gains,
        ctx.figures_dir / "fw_amp_phase_vs_tx.pdf",
    )
    cal_rows, cal = collect_calibration_rows(ctx.setup_meta)
    if cal is None:
        ctx.notes.append("No fdtd_analytic_calibration in setup_metadata.json.")
        return
    c_arr = _complex_c_from_meta(ctx.setup_meta)
    freqs = cal.get("freqs_hz") or ctx.setup_meta.get("flist_hz") or []
    if c_arr is None:
        ctx.notes.append("Calibration C(f) arrays missing from setup_metadata.json.")
        return
    _try_figure(
        ctx,
        "fw_calibration",
        save_calibration_c_figure,
        freqs,
        c_arr,
        ctx.figures_dir / "fw_calibration.pdf",
        scatter_hx_pct=cal.get("scatter_hx_pct"),
        scatter_hz_pct=cal.get("scatter_hz_pct"),
        method=str(cal.get("method") or ""),
    )
    _ = cal_rows


def write_2d_figures(ctx: ReportContext) -> None:
    run_dir = ctx.run_2d
    assert run_dir is not None
    sg_true = ctx.fwd_dir / "sg.rss"
    sg_up = latest_sg_up_file(run_dir)
    if sg_up is None or not sg_true.exists():
        if sg_up is None:
            ctx.notes.append(f"2D inverted model missing under {run_dir.name} (no sg_up.rss-*).")
        if not sg_true.exists():
            ctx.notes.append("2D true-model comparison skipped: workspace/2D/forward/sg.rss not found.")
        return
    x_t, z_t, rho_t = resistivity_from_sg_rss(sg_true)
    x_i, z_i, rho_i = resistivity_from_sg_rss(sg_up)
    label = sg_up.name
    _try_figure(
        ctx,
        "inv2d_models",
        save_2d_model_compare_figure,
        x_t,
        z_t,
        rho_t,
        x_i,
        z_i,
        rho_i,
        ctx.figures_dir / "inv2d_models.pdf",
        inv_label=label,
    )
    _try_figure(
        ctx,
        "inv2d_slices",
        save_2d_slices_figure,
        x_t,
        z_t,
        rho_t,
        x_i,
        z_i,
        rho_i,
        ctx.figures_dir / "inv2d_slices.pdf",
        inv_label=label,
    )
    pairs = available_synthetic_pairs(run_dir)
    if not pairs:
        ctx.notes.append(
            "2D observed-vs-synthetic comparison skipped: no synthetic Hx/Hz pair in the run directory."
        )
        return
    _, hx_syn, hz_syn = pairs[-1]
    hx_obs = run_dir / "Hx_data.rss"
    hz_obs = run_dir / "Hz_data.rss"
    if not hx_obs.exists() or not hz_obs.exists():
        hx_obs = ctx.cfg.inv_2d_input_dir / "Hx_data.rss"
        hz_obs = ctx.cfg.inv_2d_input_dir / "Hz_data.rss"
    wav = run_dir / "wav2d.rss"
    if not wav.exists():
        wav = ctx.fwd_dir / str(ctx.setup_meta.get("forward_wavelet") or "wav2d.rss")
    freqs = ctx.setup_meta.get("flist_hz") or []
    if not (hx_obs.exists() and hz_obs.exists() and wav.exists() and freqs):
        ctx.notes.append("2D data comparison skipped: missing observed gathers, wavelet, or frequencies.")
        return
    f_min = float(ctx.setup_meta.get("f_min_hz") or min(float(v) for v in freqs))
    n_periods = float(ctx.setup_meta.get("n_periods_extract") or 3.0)
    try:
        obs = compute_gains_for_fd_outputs(
            hx_obs, hz_obs, wav, freqs=freqs, f_min_hz=f_min, n_periods_extract=n_periods
        )
        syn = compute_gains_for_fd_outputs(
            hx_syn, hz_syn, wav, freqs=freqs, f_min_hz=f_min, n_periods_extract=n_periods
        )
    except Exception as exc:
        ctx.notes.append(f"2D data comparison skipped: {exc}")
        return
    _try_figure(
        ctx,
        "inv2d_data",
        save_obs_vs_syn_figure,
        obs,
        syn,
        ctx.figures_dir / "inv2d_data.pdf",
    )
    _try_figure(
        ctx,
        "inv2d_data_vs_tx",
        save_obs_vs_syn_vs_tx_figure,
        obs,
        syn,
        ctx.figures_dir / "inv2d_data_vs_tx.pdf",
    )


def _rebuild_1d_section(data: Mapping, summary: Mapping, sg_true: Path) -> Optional[tuple]:
    if not sg_true.exists():
        return None
    tx_x = np.asarray(data.get("tx_x", []), dtype=float)
    tx_z = np.asarray(data.get("tx_z", []), dtype=float)
    rho_layers = np.asarray(data.get("rho_layers", []), dtype=float)
    thk_layers = np.asarray(data.get("thickness_layers", []), dtype=float)
    if tx_x.size == 0 or rho_layers.size == 0 or thk_layers.size == 0:
        return None
    x_grid, z_grid, _ = resistivity_from_sg_rss(sg_true)
    background = float(summary.get("background_rho", 10.0))
    spans = []
    for i in range(thk_layers.shape[0]):
        v = np.asarray(thk_layers[i], dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            spans.append(float(np.sum(v)))
    if not spans:
        return None
    halfspan = 0.5 * float(np.median(np.asarray(spans)))
    z_start_rel = float(summary.get("z_start", -halfspan)) if summary.get("z_start") is not None else -halfspan
    z_end_rel = float(summary.get("z_end", halfspan)) if summary.get("z_end") is not None else halfspan
    order = np.argsort(tx_x)
    tx_x_s = tx_x[order]
    tx_z_s = tx_z[order] if tx_z.size else np.zeros_like(tx_x_s)
    rho_s = rho_layers[order]
    thk_s = thk_layers[order]
    n_tx = int(rho_s.shape[0])
    mean_profiles = np.full((n_tx, z_grid.size), background, dtype=float)
    for i in range(n_tx):
        rho_row = np.asarray(rho_s[i], dtype=float)
        thk_row = np.asarray(thk_s[i], dtype=float)
        mask = np.isfinite(rho_row)
        rho_valid = rho_row[mask]
        thk_valid = thk_row[np.isfinite(thk_row)]
        if rho_valid.size == 0:
            continue
        n_layers = int(rho_valid.size)
        thk_valid = thk_valid[: max(n_layers - 1, 0)]
        z_top = float(tx_z_s[i]) + z_start_rel
        z_bottom = float(tx_z_s[i]) + z_end_rel
        interfaces = float(tx_z_s[i]) + (z_start_rel + np.cumsum(thk_valid)) if thk_valid.size else np.array([])
        z0 = z_top
        for li in range(n_layers):
            if li < interfaces.size:
                z1 = float(interfaces[li])
                m = (z_grid >= z0) & (z_grid < z1)
            else:
                z1 = z_bottom
                m = (z_grid >= z0) & (z_grid <= z1)
            mean_profiles[i, m] = float(rho_valid[li])
            z0 = z1
    section = np.zeros((z_grid.size, x_grid.size), dtype=float)
    for iz in range(z_grid.size):
        section[iz, :] = np.interp(x_grid, tx_x_s, mean_profiles[:, iz], left=background, right=background)
    return x_grid, z_grid, section, None


def write_1d_figures(ctx: ReportContext, data: Mapping, summary: Mapping) -> None:
    sg_true = ctx.fwd_dir / "sg.rss"
    true_x = true_z = true_rho = None
    if sg_true.exists():
        true_x, true_z, true_rho = resistivity_from_sg_rss(sg_true)
    sec_x = data.get("section_x")
    sec_z = data.get("section_z")
    sec_rho = data.get("section_rho")
    sec_std = data.get("section_rho_std")
    if sec_x is None or sec_z is None or sec_rho is None:
        rebuilt = _rebuild_1d_section(data, summary, sg_true)
        if rebuilt is not None:
            sec_x, sec_z, sec_rho, sec_std = rebuilt
    if sec_x is not None and sec_z is not None and sec_rho is not None:
        _try_figure(
            ctx,
            "inv1d_section",
            save_1d_section_figure,
            np.asarray(sec_x, dtype=float),
            np.asarray(sec_z, dtype=float),
            np.asarray(sec_rho, dtype=float),
            ctx.figures_dir / "inv1d_section.pdf",
            sec_std=None if sec_std is None else np.asarray(sec_std, dtype=float),
            true_x=true_x,
            true_z=true_z,
            true_rho=true_rho,
        )
    else:
        ctx.notes.append("1D pseudo-2D section skipped: no section arrays and rebuild failed.")

    tx_ids = np.asarray(data.get("tx_ids", []), dtype=int)
    if tx_ids.size and "rho_layers" in data:
        _try_figure(
            ctx,
            "inv1d_rho_depth",
            save_1d_rho_vs_depth_figure,
            tx_ids,
            np.asarray(data.get("tx_x", []), dtype=float),
            np.asarray(data["rho_layers"], dtype=float),
            np.asarray(data.get("thickness_layers", []), dtype=float),
            ctx.figures_dir / "inv1d_rho_depth.pdf",
            z_start=summary.get("z_start"),
            z_end=summary.get("z_end"),
            tx_z=np.asarray(data.get("tx_z", []), dtype=float) if "tx_z" in data else None,
        )
    if tx_ids.size:
        mid = tx_ids[tx_ids.size // 2]
        obs_hx = data.get(f"obs_hx_gain_tx{int(mid)}")
        obs_hz = data.get(f"obs_hz_gain_tx{int(mid)}")
        pred_hx = data.get(f"pred_hxh_mean_tx{int(mid)}")
        pred_hz = data.get(f"pred_hxhz_mean_tx{int(mid)}")
        freqs = data.get(f"freqs_tx{int(mid)}", summary.get("freqs_hz"))
        rx = data.get(f"rx_x_tx{int(mid)}")
        if obs_hx is not None and pred_hx is not None and freqs is not None:
            _try_figure(
                ctx,
                "inv1d_data",
                save_1d_obs_pred_figure,
                freqs,
                obs_hx,
                obs_hz if obs_hz is not None else np.full_like(obs_hx, np.nan),
                pred_hx,
                pred_hz if pred_hz is not None else np.full_like(pred_hx, np.nan),
                ctx.figures_dir / "inv1d_data.pdf",
                tx_id=int(mid),
                rx=rx,
                c_per_freq=_complex_c_from_meta(ctx.setup_meta),
            )
            _try_figure(
                ctx,
                "inv1d_data_vs_tx",
                save_1d_obs_pred_vs_tx_figure,
                data,
                ctx.figures_dir / "inv1d_data_vs_tx.pdf",
                freqs=freqs,
                c_per_freq=_complex_c_from_meta(ctx.setup_meta),
            )
        else:
            ctx.notes.append("1D observed-vs-predicted figure skipped: missing gain arrays in NPZ.")
    if "chi2" in data or "misfit" in data:
        _try_figure(
            ctx,
            "inv1d_chi2",
            save_1d_chi2_figure,
            tx_ids,
            data.get("chi2", data.get("misfit")),
            ctx.figures_dir / "inv1d_chi2.pdf",
            misfit=data.get("misfit") if "chi2" in data else None,
            tx_x=data.get("tx_x"),
        )


def _include_fig(ctx: ReportContext, key: str, caption: str) -> str:
    path = ctx.figures.get(key)
    if path is None:
        return ""
    rel = path.name
    return _figure_block(rel, caption)


def render_tex(ctx: ReportContext) -> str:
    included = []
    if ctx.run_2d is not None:
        included.append(f"2D inversion {ctx.run_2d.name}")
    if ctx.run_1d is not None:
        included.append(f"1D inversion {ctx.run_1d.name}")
    included_txt = ", ".join(included) if included else "forward setup only"
    lines = [
        r"% EM inversion workshop workflow report",
        r"% Auto-generated; do not edit by hand.",
        r"\PassOptionsToPackage{hyphens}{url}",
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\usepackage{hyperref}",
        r"\usepackage{booktabs}",
        r"\usepackage{parskip}",
        r"\usepackage{graphicx}",
        r"\usepackage{flafter}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\graphicspath{{figures/}}",
        r"\title{EM Inversion Workshop\\Workflow Report}",
        r"\author{}",
        rf"\date{{{latex_escape(ctx.timestamp)}}}",
        r"\begin{document}",
        r"\maketitle",
        (
            "This report snapshots the current workspace: forward-modelling setup, "
            "modelled data, and any inversion runs that were present when the script ran. "
            "It does not re-run modelling or inversion."
        ),
        "",
        r"\section{Provenance}",
        _kv_table(
            [
                ("Workshop root", ctx.root),
                ("Generated", ctx.timestamp),
                ("Included", included_txt),
                ("setup_metadata.json", _display_path(ctx.setup_meta_path, ctx.root)),
                ("2D run", None if ctx.run_2d is None else ctx.run_2d.name),
                ("1D run", None if ctx.run_1d is None else ctx.run_1d.name),
            ]
        ),
        "",
        r"\section{Forward setup}",
    ]
    for title, rows in collect_fw_rows(ctx):
        lines.append(rf"\subsection{{{latex_escape(title)}}}")
        lines.append(_kv_table(rows))
        lines.append("")
    lines.append(r"\subsection{Model and wavelet}")
    block = _include_fig(ctx, "fw_resistivity", "Forward resistivity model with transmitter and receiver locations.")
    lines.append(block or _note_block("Resistivity model figure not available."))
    block = _include_fig(ctx, "fw_wavelet", "Source wavelet in time and frequency.")
    lines.append(block or _note_block("Wavelet figure not available."))

    lines.append(r"\section{Modelled data}")
    block = _include_fig(
        ctx,
        "fw_amp_phase",
        "Steady-state channel-gain amplitude and phase versus local receiver index for a mid-line transmitter. Colours are frequencies.",
    )
    lines.append(block or _note_block("Modelled amp/phase vs-rx figure not available."))
    block = _include_fig(
        ctx,
        "fw_amp_phase_vs_tx",
        "Steady-state channel-gain amplitude and phase versus transmitter index, with amplitude and phase rows for each receiver. Colours are frequencies.",
    )
    lines.append(block or _note_block("Modelled amp/phase vs-Tx figure not available."))
    cal_rows, cal = collect_calibration_rows(ctx.setup_meta)
    if cal:
        lines.append(r"\subsection{FDTD--analytic calibration}")
        lines.append(_kv_table(cal_rows))
        block = _include_fig(
            ctx,
            "fw_calibration",
            "Global FDTD--analytic scale C(f) stored by Step 02.",
        )
        lines.append(block or _note_block("Calibration C(f) figure not available."))
    else:
        lines.append(_note_block("No FDTD--analytic calibration is stored in setup_metadata.json."))

    if ctx.run_2d is not None:
        lines.append(r"\section{2D inversion}")
        lines.append(rf"Results from \texttt{{{latex_escape(ctx.run_2d.name)}}}.")
        lines.append(r"\subsection{Parameters}")
        lines.append(_kv_table(collect_2d_inv_rows(ctx)))
        lines.append(r"\subsection{Final model}")
        block = _include_fig(ctx, "inv2d_models", "True resistivity model versus the latest inverted conductivity update (shown as resistivity).")
        lines.append(block or _note_block("2D inverted-model figure not available."))
        block = _include_fig(ctx, "inv2d_slices", "Resistivity slices through the true and inverted 2D models.")
        lines.append(block or "")
        lines.append(r"\subsection{Data comparison}")
        block = _include_fig(
            ctx,
            "inv2d_data",
            "Observed versus already-generated synthetic Hx/Hz channel gains versus receiver index for a mid-line transmitter.",
        )
        lines.append(block or _note_block("No synthetic Hx/Hz pair was found in the 2D run directory, so the data comparison was omitted."))
        block = _include_fig(
            ctx,
            "inv2d_data_vs_tx",
            "Observed versus synthetic Hx/Hz channel gains versus transmitter index, with amplitude and phase rows for each receiver.",
        )
        lines.append(block or "")
    else:
        lines.append(r"\section{2D inversion}")
        lines.append(_note_block("No 2D inversion run was included."))

    if ctx.run_1d is not None:
        summary_path = ctx.run_1d / "analytic_1d_inversion_summary.json"
        meta_path = ctx.run_1d / "run_metadata.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        run_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        lines.append(r"\section{1D inversion}")
        lines.append(rf"Results from \texttt{{{latex_escape(ctx.run_1d.name)}}}.")
        lines.append(r"\subsection{Parameters}")
        lines.append(_kv_table(collect_1d_inv_rows(summary, run_meta)))
        lines.append(r"\subsection{Final model}")
        block = _include_fig(
            ctx,
            "inv1d_section",
            "Pseudo-2D section from independent per-transmitter 1D inversions, with the true model when available.",
        )
        lines.append(block or _note_block("1D pseudo-2D section not available."))
        block = _include_fig(ctx, "inv1d_rho_depth", "Layered 1D resistivity versus depth for representative transmitters.")
        lines.append(block or "")
        lines.append(r"\subsection{Data comparison}")
        block = _include_fig(
            ctx,
            "inv1d_data",
            "Observed FDTD channel gains versus analytic predictions of the inverted 1D model, versus receiver (C(f) applied when stored).",
        )
        lines.append(block or _note_block("1D observed-versus-predicted figure not available."))
        block = _include_fig(
            ctx,
            "inv1d_data_vs_tx",
            "Observed versus predicted 1D channel gains versus transmitter, with amplitude and phase rows for each receiver.",
        )
        lines.append(block or "")
        block = _include_fig(ctx, "inv1d_chi2", "Per-transmitter chi-squared and misfit.")
        lines.append(block or "")
    else:
        lines.append(r"\section{1D inversion}")
        lines.append(_note_block("No 1D inversion run was included."))

    if ctx.notes:
        lines.append(r"\section{Notes}")
        lines.append(r"\begin{itemize}")
        for note in ctx.notes:
            lines.append(rf"\item {latex_escape(note)}")
        lines.append(r"\end{itemize}")

    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


def compile_pdf(tex_path: Path) -> Path:
    exe = shutil.which("pdflatex")
    if exe is None:
        raise FileNotFoundError("pdflatex not found on PATH")
    tex_path = Path(tex_path)
    for _ in range(2):
        proc = subprocess.run(
            [exe, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            lines = (proc.stdout or proc.stderr or "").splitlines()
            err_lines = [ln for ln in lines if ln.startswith("!") or "Error" in ln]
            tail = "\n".join(err_lines[-12:] or lines[-20:])
            raise RuntimeError(f"pdflatex failed:\n{tail}")
    return tex_path.with_suffix(".pdf")


def build_report(
    *,
    root: Optional[Path] = None,
    include_2d: bool = True,
    include_1d: bool = True,
    run_2d: Optional[str] = None,
    run_1d: Optional[str] = None,
    compile_pdf_flag: bool = False,
) -> dict[str, Any]:
    root = (root or Path.cwd()).resolve()
    cfg = load_config(root)
    meta_path = setup_metadata_path(root)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Finalize Step 01 (Generate FD inputs) before making a report."
        )
    meta = load_setup_metadata(root=root, path=meta_path)
    if not meta:
        raise FileNotFoundError(f"Could not read setup metadata at {meta_path}")

    report_dir = cfg.workspace / "report"
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ctx = ReportContext(
        root=root,
        cfg=cfg,
        fwd_dir=cfg.fwd_2d_dir,
        setup_meta=meta,
        setup_meta_path=meta_path,
        report_dir=report_dir,
        figures_dir=figures_dir,
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    if include_2d:
        ctx.run_2d = resolve_run_dir(cfg.inv_2d_runs_dir, run_2d, kind="2D", required=bool(run_2d))
    if include_1d:
        ctx.run_1d = resolve_run_dir(cfg.inv_1d_runs_dir, run_1d, kind="1D", required=bool(run_1d))

    write_fw_figures(ctx)
    write_modelled_data_figures(ctx)
    if ctx.run_2d is not None:
        write_2d_figures(ctx)
    if ctx.run_1d is not None:
        npz_path = ctx.run_1d / "analytic_1d_inversion_results.npz"
        summary_path = ctx.run_1d / "analytic_1d_inversion_summary.json"
        if npz_path.exists():
            data = dict(np.load(npz_path, allow_pickle=True))
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            write_1d_figures(ctx, data, summary)
        else:
            ctx.notes.append(f"1D result NPZ not found: {npz_path}")

    tex_path = report_dir / "workflow_report.tex"
    tex_path.write_text(render_tex(ctx))
    pdf_path = None
    compile_error = None
    if compile_pdf_flag:
        try:
            pdf_path = compile_pdf(tex_path)
        except Exception as exc:
            compile_error = str(exc)
    return {
        "tex_path": tex_path,
        "pdf_path": pdf_path,
        "report_dir": report_dir,
        "figures": dict(ctx.figures),
        "notes": list(ctx.notes),
        "run_2d": ctx.run_2d,
        "run_1d": ctx.run_1d,
        "compile_error": compile_error,
    }


__all__ = [
    "ReportContext",
    "available_synthetic_pairs",
    "build_report",
    "compile_pdf",
    "latest_sg_up_file",
    "list_run_dirs",
    "resolve_run_dir",
]
