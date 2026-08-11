"""Defaults for workshop notebooks from ``setup_metadata.json`` (notebook 01).

Shared parameters (frequencies, n_periods, eps_r, resistivity bounds) must be
read here rather than hardcoded in later notebooks. Numeric fallbacks exist only
for a pristine tree before notebook 01 has been finalized.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from scripts.modules.workshop_config import load_config

FALLBACK_FREQUENCIES_HZ = [2000.0, 4000.0, 6000.0]
FALLBACK_N_PERIODS_EXTRACT = 3.0
FALLBACK_EPS_R = 7.0
FALLBACK_RHO_MIN = 1.0
FALLBACK_RHO_MAX = 200.0


def setup_metadata_path(root: Path | None = None) -> Path:
    cfg = load_config(root)
    return cfg.fwd_2d_dir / "setup_metadata.json"


def load_setup_metadata(root: Path | None = None, path: Path | str | None = None) -> dict:
    p = Path(path) if path is not None else setup_metadata_path(root)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def meta_available(root: Path | None = None, path: Path | str | None = None) -> bool:
    meta = load_setup_metadata(root=root, path=path)
    return bool(meta)


def default_frequencies(
    root: Path | None = None,
    path: Path | str | None = None,
    fallback: Optional[Sequence[float]] = None,
) -> list[float]:
    meta = load_setup_metadata(root=root, path=path)
    vals = meta.get("flist_hz") or meta.get("freqs_hz") or []
    out = [float(v) for v in vals]
    if out:
        return out
    return [float(v) for v in (fallback if fallback is not None else FALLBACK_FREQUENCIES_HZ)]


def default_n_periods_extract(
    root: Path | None = None,
    path: Path | str | None = None,
    fallback: float = FALLBACK_N_PERIODS_EXTRACT,
) -> float:
    meta = load_setup_metadata(root=root, path=path)
    v = meta.get("n_periods_extract")
    if v is None:
        # Older meta may only store wavelet n_periods under wavelet_params.
        wp = meta.get("wavelet_params") or {}
        v = wp.get("n_periods")
    return float(v) if v is not None else float(fallback)


def default_f_min_hz(
    root: Path | None = None,
    path: Path | str | None = None,
    freqs: Optional[Sequence[float]] = None,
) -> float:
    meta = load_setup_metadata(root=root, path=path)
    v = meta.get("f_min_hz")
    if v is not None:
        return float(v)
    freq_list = list(freqs) if freqs is not None else default_frequencies(root=root, path=path)
    if freq_list:
        return float(min(freq_list))
    return float(min(FALLBACK_FREQUENCIES_HZ))


def default_eps_r(
    root: Path | None = None,
    path: Path | str | None = None,
    fallback: float = FALLBACK_EPS_R,
) -> float:
    meta = load_setup_metadata(root=root, path=path)
    for key in ("eps_r_used", "eps_r"):
        if key in meta and meta[key] is not None:
            return float(meta[key])
    return float(fallback)


def default_rho_bounds(
    root: Path | None = None,
    path: Path | str | None = None,
) -> Tuple[float, float]:
    meta = load_setup_metadata(root=root, path=path)
    rho_min = meta.get("rho_min_ohm_m")
    rho_max = meta.get("rho_max_ohm_m")
    lo = float(rho_min) if rho_min is not None else FALLBACK_RHO_MIN
    hi = float(rho_max) if rho_max is not None else FALLBACK_RHO_MAX
    if hi <= lo:
        hi = max(lo * 10.0, FALLBACK_RHO_MAX)
    return lo, hi


def default_background_rho(
    root: Path | None = None,
    path: Path | str | None = None,
) -> float:
    """Seed background resistivity from setup rho_min (homogeneous host)."""
    lo, _ = default_rho_bounds(root=root, path=path)
    return float(lo)


def format_freq_list(freqs: Sequence[float]) -> str:
    return ",".join(f"{float(v):g}" for v in freqs)


def status_note_if_meta_missing(root: Path | None = None, path: Path | str | None = None) -> Optional[str]:
    if meta_available(root=root, path=path):
        return None
    p = Path(path) if path is not None else setup_metadata_path(root)
    return (
        f"No setup_metadata.json at {p} — using built-in fallbacks. "
        "Finalize notebook 01 so later notebooks remember frequencies and design parameters."
    )


__all__ = [
    "FALLBACK_FREQUENCIES_HZ",
    "FALLBACK_N_PERIODS_EXTRACT",
    "FALLBACK_EPS_R",
    "FALLBACK_RHO_MIN",
    "FALLBACK_RHO_MAX",
    "default_background_rho",
    "default_eps_r",
    "default_f_min_hz",
    "default_frequencies",
    "default_n_periods_extract",
    "default_rho_bounds",
    "format_freq_list",
    "load_setup_metadata",
    "meta_available",
    "setup_metadata_path",
    "status_note_if_meta_missing",
]
