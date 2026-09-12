"""DE population envelope + DE→BlockInv hybrid 1D inversion (Step 05 default path)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from scripts.modules.inversion_1d import (
    DEFAULT_COMPONENTS,
    n_tensor_data,
    tensor_objective,
    unpack_model_params,
)
from scripts.modules.inversion_blockinv import polish_de_with_blockinv
from scripts.modules.inversion_tuning import run_de_population

HYBRID_OPTIMIZER_ID = "de_blockinv_hybrid"
DEFAULT_ENVELOPE_PERCENTILES = (10, 50, 90)
DEFAULT_ENVELOPE_NZ = 300

# Fixed defaults matching ``scripts/experiments/fast_ambiguity.py`` (~5–8 s/Tx).
HYBRID_DEFAULTS: Dict[str, Any] = {
    "optimizer": HYBRID_OPTIMIZER_ID,
    "popsize": 12,
    "maxiter": 8,
    "reg_lambda": 500.0,
    "block_max_iter": 8,
    "seed": 42,
    "n_jobs": -1,
    "n_runs": 1,
}
HYBRID_COMPONENT_WEIGHTS = {f"w_{c}": 1.0 for c in ("Cxx", "Cxz", "Czx", "Czz")}


def apply_hybrid_defaults(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Attach fixed hybrid solver settings; user cfg supplies model/bounds only."""
    out = dict(cfg)
    out.update(HYBRID_DEFAULTS)
    out.update(HYBRID_COMPONENT_WEIGHTS)
    out["optimizer"] = HYBRID_OPTIMIZER_ID
    return out


def score_chi2_data(
    params,
    tx_entry,
    cfg: Mapping[str, Any],
    cal,
    *,
    components=DEFAULT_COMPONENTS,
) -> float:
    """Reduced χ² with reg_lambda=0 (data term only)."""
    mis = tensor_objective(
        params,
        tx_entry=tx_entry,
        n_layers=int(cfg["n_layers"]),
        z_start_rel=float(cfg["z_start_rel"]),
        z_end_rel=float(cfg["z_end_rel"]),
        eps_r=cfg["eps_r"],
        reg_lambda=0.0,
        cal=cal,
        components=tuple(components),
        n_nodes=int(cfg.get("n_nodes", 120)),
        snap_dz=cfg.get("snap_dz"),
        snap_origin_m=cfg.get("snap_origin_m"),
    )
    return float(mis / max(n_tensor_data(tx_entry, tuple(components)), 1))


def log10_rho_on_grid(params, cfg, z_grid: np.ndarray) -> np.ndarray:
    """Stairstep log10 ρ sampled on a Tx-relative depth grid."""
    rho, _thk, depth = unpack_model_params(
        params,
        int(cfg["n_layers"]),
        float(cfg["z_start_rel"]),
        float(cfg["z_end_rel"]),
    )
    rho = np.asarray(rho, dtype=float)
    idx = np.searchsorted(np.asarray(depth, dtype=float), z_grid)
    idx = np.clip(idx, 0, rho.size - 1)
    return np.log10(np.maximum(rho[idx], 1e-12))


def population_depth_envelope(
    population_rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    z_grid: Optional[np.ndarray] = None,
    percentiles: Tuple[float, float, float] = DEFAULT_ENVELOPE_PERCENTILES,
) -> Dict[str, np.ndarray]:
    """Depth-resolved p_lo / p_mid / p_hi from DE population members."""
    z_start = float(cfg["z_start_rel"])
    z_end = float(cfg["z_end_rel"])
    z_grid = np.asarray(
        z_grid if z_grid is not None else np.linspace(z_start, z_end, DEFAULT_ENVELOPE_NZ),
        dtype=float,
    )
    if not population_rows:
        nan = np.full(z_grid.size, np.nan)
        return {"z": z_grid, "rho_p10": nan, "rho_p50": nan, "rho_p90": nan}
    stack = np.vstack(
        [log10_rho_on_grid(r["params"], cfg, z_grid) for r in population_rows]
    )
    lo, mid, hi = np.percentile(stack, list(percentiles), axis=0)
    return {
        "z": z_grid,
        "rho_p10": lo,
        "rho_p50": mid,
        "rho_p90": hi,
        "p_lo": lo,
        "p_mid": mid,
        "p_hi": hi,
    }


def invert_tx_hybrid_population(
    tx_entry: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    cal,
    seed: Optional[int] = None,
    score_chi2_fn: Optional[Callable[..., float]] = None,
    envelope_percentiles: Tuple[float, float, float] = DEFAULT_ENVELOPE_PERCENTILES,
    setup_meta_path=None,
) -> Dict[str, Any]:
    """One Tx: DE final population → envelope; DE best → BlockInv hybrid point model."""
    cfg = dict(cfg)
    cfg["optimizer"] = HYBRID_OPTIMIZER_ID
    seed = int(cfg.get("seed") if seed is None else seed)
    components = tuple(cfg.get("components") or DEFAULT_COMPONENTS)
    chi2_fn = score_chi2_fn or (
        lambda p, tx, c, calibration: score_chi2_data(p, tx, c, calibration, components=components)
    )

    de = run_de_population(cfg, tx_entry, seed)
    population_rows = []
    population_chi2 = []
    for params in de["population"]:
        p = np.asarray(params, dtype=float)
        c2 = chi2_fn(p, tx_entry, cfg, cal)
        population_rows.append({"params": p, "chi2": c2})
        population_chi2.append(c2)

    best_idx = int(np.argmin(population_chi2)) if population_chi2 else 0
    best_params = np.asarray(population_rows[best_idx]["params"], dtype=float)
    de_best_chi2 = float(population_chi2[best_idx]) if population_chi2 else float(de["chi2_data"])

    hybrid = polish_de_with_blockinv(
        tx_entry,
        cfg,
        best_params,
        de_chi2=de_best_chi2,
        de_data_misfit=float(de.get("data_misfit", de_best_chi2 * max(n_tensor_data(tx_entry, components), 1))),
        setup_meta_path=setup_meta_path or cfg.get("setup_meta_path"),
    )

    envelope = population_depth_envelope(
        population_rows,
        cfg,
        percentiles=envelope_percentiles,
    )

    tx_z = float(tx_entry.get("tx_z", 0.0))
    _, _, depth_rel = unpack_model_params(
        hybrid["params"],
        int(cfg["n_layers"]),
        float(cfg["z_start_rel"]),
        float(cfg["z_end_rel"]),
    )
    depth_rel = np.asarray(depth_rel, dtype=float)
    depth_abs = tx_z + depth_rel

    return {
        "success": bool(de.get("success", True)) and bool(hybrid.get("success", True)),
        "message": str(hybrid.get("message", "")),
        "optimizer": HYBRID_OPTIMIZER_ID,
        "misfit": float(hybrid.get("misfit", np.nan)),
        "data_misfit": float(hybrid.get("data_misfit", np.nan)),
        "reg_norm": float(de.get("reg_norm", 0.0)),
        "chi2": float(hybrid.get("chi2", np.nan)),
        "de_best_chi2": de_best_chi2,
        "blockinv_chi2": float(hybrid.get("blockinv_chi2", np.nan)),
        "hybrid_polish_reverted": bool(hybrid.get("polish_reverted", False)),
        "params": np.asarray(hybrid["params"], dtype=float),
        "rho": np.asarray(hybrid["rho"], dtype=float),
        "thickness": np.asarray(hybrid["thickness"], dtype=float),
        "depth": depth_abs,
        "depth_rel": depth_rel,
        "z_top": float(hybrid.get("z_top", tx_z + float(cfg["z_start_rel"]))),
        "z_bottom": float(hybrid.get("z_bottom", tx_z + float(cfg["z_end_rel"]))),
        "pred": hybrid.get("pred"),
        "pred_hxh": hybrid.get("pred_hxh"),
        "pred_hxhz": hybrid.get("pred_hxhz"),
        "rho_unc": np.asarray(hybrid.get("rho_unc", []), dtype=float),
        "thickness_unc": np.asarray(hybrid.get("thickness_unc", []), dtype=float),
        "depth_unc": np.asarray(hybrid.get("depth_unc", []), dtype=float),
        "de_population": np.asarray(de["population"], dtype=float),
        "population_chi2": np.asarray(population_chi2, dtype=float),
        "population_rows": population_rows,
        "envelope": envelope,
        "envelope_percentiles": list(envelope_percentiles),
        "n_envelope": len(population_rows),
        "de_nfev": int(de.get("nfev", -1)),
        "de_nit": int(de.get("nit", -1)),
        "seed": seed,
    }


__all__ = [
    "HYBRID_OPTIMIZER_ID",
    "HYBRID_DEFAULTS",
    "HYBRID_COMPONENT_WEIGHTS",
    "apply_hybrid_defaults",
    "DEFAULT_ENVELOPE_PERCENTILES",
    "score_chi2_data",
    "log10_rho_on_grid",
    "population_depth_envelope",
    "invert_tx_hybrid_population",
]
