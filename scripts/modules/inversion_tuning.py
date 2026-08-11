"""Single-Tx DE budget and Tikhonov-lambda L-curve tuners for notebook 05.

DE budget tuner raises (popsize, maxiter) until multi-seed total objectives agree.
Lambda tuner sweeps reg_lambda at fixed DE settings and picks the L-curve corner.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from scripts.modules.analytic_1d_forward import ForwardRejected, forward_1d_gains

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = None  # type: ignore
    delayed = None  # type: ignore

# Cheapest -> costliest DE budgets (SciPy popsize is a multiplier: pop = popsize * ndim).
DEFAULT_DE_BUDGETS: Tuple[Tuple[int, int], ...] = (
    (10, 30),
    (15, 100),
    (15, 200),
    (20, 200),
    (20, 300),
)

DEFAULT_LAMBDA_GRID: Tuple[float, ...] = (
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    500.0,
    1000.0,
    3000.0,
    1.0e4,
    3.0e4,
    1.0e5,
)

DEFAULT_SEED_SPREAD_TOL = 0.05
DEFAULT_N_TUNE_SEEDS = 5
_REJECT_COST = 1e12


def unpack_model_params(params, n_layers, z_start_rel, z_end_rel):
    """Match notebook 05 parameterization (log10 rho / log10 thk, span-normalized)."""
    p = np.asarray(params, dtype=float)
    lrho = p[:n_layers]
    rho = np.power(10.0, lrho)
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


def build_bounds(n_layers, log10_rho_min, log10_rho_max, log10_thk_min, log10_thk_max):
    b = [(float(log10_rho_min), float(log10_rho_max)) for _ in range(int(n_layers))]
    for _ in range(int(n_layers) - 1):
        b.append((float(log10_thk_min), float(log10_thk_max)))
    return b


def _forward_analytic_for_tx(params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r, n_nodes=120):
    rho, thk, _ = unpack_model_params(params, n_layers, z_start_rel, z_end_rel)
    tx_z = float(tx_entry["tx_z"])
    off_x = np.asarray(tx_entry["off_x"], dtype=float)
    off_z = np.asarray(tx_entry["off_z"], dtype=float)
    rx_depth_m = tx_z + float(off_z[0]) if off_z.size else tx_z
    return forward_1d_gains(rho, thk, tx_entry["freqs"], off_x, tx_z, rx_depth_m, eps_r, n_nodes=n_nodes)


def split_objective(
    params,
    tx_entry,
    n_layers,
    z_start_rel,
    z_end_rel,
    eps_r,
    reg_lambda,
    w_hxh,
    w_hxhz,
    sigma_hx,
    sigma_hz,
    C=None,
) -> Tuple[float, float, float]:
    """Return (data_misfit, reg_norm, total) matching notebook 05 complex_gain_objective."""
    try:
        hx_pred, hz_pred = _forward_analytic_for_tx(
            params, tx_entry, n_layers, z_start_rel, z_end_rel, eps_r,
        )
    except ForwardRejected:
        return _REJECT_COST, 0.0, _REJECT_COST

    obs_hx = np.asarray(tx_entry["obs_hx_gain"], dtype=complex)
    obs_hz = np.asarray(tx_entry["obs_hz_gain"], dtype=complex)
    cal = np.asarray(C, dtype=complex)[:, None] if C is not None else 1.0

    res_x = (cal * hx_pred - obs_hx) / np.maximum(np.asarray(sigma_hx, dtype=float)[:, None], 1e-300)
    res_z = (cal * hz_pred - obs_hz) / np.maximum(np.asarray(sigma_hz, dtype=float)[:, None], 1e-300)

    m1, m2 = np.isfinite(res_x), np.isfinite(res_z)
    if not np.any(m1) and not np.any(m2):
        return _REJECT_COST, 0.0, _REJECT_COST

    data_misfit = 0.0
    if np.any(m1):
        data_misfit += float(w_hxh) * float(np.nansum(np.where(m1, np.abs(res_x) ** 2, np.nan)))
    if np.any(m2):
        data_misfit += float(w_hxhz) * float(np.nansum(np.where(m2, np.abs(res_z) ** 2, np.nan)))

    reg_norm = 0.0
    if int(n_layers) > 1:
        lrho = np.asarray(params[:n_layers], dtype=float)
        reg_norm = float(np.mean(np.diff(lrho) ** 2))

    total = data_misfit + float(reg_lambda) * reg_norm
    return data_misfit, reg_norm, total


def _n_data_from_tx(tx_entry) -> int:
    nfreq = int(np.asarray(tx_entry["freqs"]).size)
    nrx = int(np.asarray(tx_entry["off_x"]).size)
    return 2 * nfreq * nrx * 2  # Hx+Hz, real+imag


def relative_objective_spread(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.inf
    med = float(np.median(arr))
    if med <= 0.0:
        return np.inf
    return float((np.max(arr) - np.min(arr)) / med)


def run_de_once(cfg: Mapping[str, Any], tx_entry: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    """One DE inversion; returns params and split objective terms."""
    cal = cfg["calibration"]
    n_layers = int(cfg["n_layers"])
    bounds = build_bounds(
        n_layers,
        cfg["log10_rho_min"],
        cfg["log10_rho_max"],
        cfg["log10_thk_min"],
        cfg["log10_thk_max"],
    )

    def obj(p):
        _data, _reg, total = split_objective(
            p,
            tx_entry=tx_entry,
            n_layers=n_layers,
            z_start_rel=cfg["z_start_rel"],
            z_end_rel=cfg["z_end_rel"],
            eps_r=cfg["eps_r"],
            reg_lambda=cfg["reg_lambda"],
            w_hxh=cfg["w_hxh"],
            w_hxhz=cfg["w_hxhz"],
            sigma_hx=cal["sigma_hx"],
            sigma_hz=cal["sigma_hz"],
            C=cal.get("C"),
        )
        return total

    out = differential_evolution(
        obj,
        bounds=bounds,
        maxiter=int(cfg["maxiter"]),
        popsize=int(cfg["popsize"]),
        seed=int(seed),
        polish=False,
        workers=1,
        updating="deferred",
    )
    best = np.asarray(out.x, dtype=float)
    data_misfit, reg_norm, total = split_objective(
        best,
        tx_entry=tx_entry,
        n_layers=n_layers,
        z_start_rel=cfg["z_start_rel"],
        z_end_rel=cfg["z_end_rel"],
        eps_r=cfg["eps_r"],
        reg_lambda=cfg["reg_lambda"],
        w_hxh=cfg["w_hxh"],
        w_hxhz=cfg["w_hxhz"],
        sigma_hx=cal["sigma_hx"],
        sigma_hz=cal["sigma_hz"],
        C=cal.get("C"),
    )
    n_data = _n_data_from_tx(tx_entry)
    return {
        "success": bool(getattr(out, "success", True)),
        "seed": int(seed),
        "params": best,
        "total": float(total),
        "data_misfit": float(data_misfit),
        "reg_norm": float(reg_norm),
        "chi2_data": float(data_misfit / max(n_data, 1)),
        "popsize": int(cfg["popsize"]),
        "maxiter": int(cfg["maxiter"]),
        "reg_lambda": float(cfg["reg_lambda"]),
        "nit": int(getattr(out, "nit", -1)),
        "nfev": int(getattr(out, "nfev", -1)),
    }


def _parallel_map(fn, items: Sequence[Any], n_jobs: int) -> List[Any]:
    if not items:
        return []
    if Parallel is None or delayed is None:
        return [fn(*item) if isinstance(item, tuple) else fn(item) for item in items]
    return list(
        Parallel(n_jobs=n_jobs)(
            delayed(fn)(*item) if isinstance(item, tuple) else delayed(fn)(item)
            for item in items
        )
    )


def _cfg_with_budget(cfg: Mapping[str, Any], popsize: int, maxiter: int, seed: int) -> dict:
    out = dict(cfg)
    out["popsize"] = int(popsize)
    out["maxiter"] = int(maxiter)
    out["seed"] = int(seed)
    out["optimizer"] = "differential_evolution"
    return out


def _run_de_with_lambda(
    cfg: Mapping[str, Any],
    tx_entry: Mapping[str, Any],
    lam: float,
    seed: int,
) -> Dict[str, Any]:
    """Top-level helper for joblib (must be picklable)."""
    cfg_k = dict(cfg)
    cfg_k["reg_lambda"] = float(lam)
    cfg_k["seed"] = int(seed)
    cfg_k["optimizer"] = "differential_evolution"
    return run_de_once(cfg_k, tx_entry, int(seed))


def tune_de_budget(
    cfg: Mapping[str, Any],
    tx_entry: Mapping[str, Any],
    *,
    budgets: Sequence[Tuple[int, int]] = DEFAULT_DE_BUDGETS,
    n_seeds: int = DEFAULT_N_TUNE_SEEDS,
    base_seed: Optional[int] = None,
    spread_tol: float = DEFAULT_SEED_SPREAD_TOL,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Raise DE budget until multi-seed total-objective spread is within tol."""
    if "calibration" not in cfg:
        raise ValueError("cfg must include calibration from notebook 02.")
    base_seed = int(cfg.get("seed", 42) if base_seed is None else base_seed)
    n_seeds = max(int(n_seeds), 1)
    seeds = [base_seed + k for k in range(n_seeds)]

    stages: List[Dict[str, Any]] = []
    recommended = None
    warning = None

    for popsize, maxiter in budgets:
        jobs = [
            (_cfg_with_budget(cfg, popsize, maxiter, seed), tx_entry, seed)
            for seed in seeds
        ]
        runs = _parallel_map(run_de_once, jobs, n_jobs=int(n_jobs))
        totals = [float(r["total"]) for r in runs]
        chi2s = [float(r["chi2_data"]) for r in runs]
        spread = relative_objective_spread(totals)
        stable = bool(spread <= float(spread_tol))
        stage = {
            "popsize": int(popsize),
            "maxiter": int(maxiter),
            "seeds": list(seeds),
            "runs": runs,
            "total_mean": float(np.nanmean(totals)),
            "total_std": float(np.nanstd(totals)),
            "total_spread": float(spread),
            "chi2_data_mean": float(np.nanmean(chi2s)),
            "chi2_data_std": float(np.nanstd(chi2s)),
            "stable": stable,
        }
        stages.append(stage)
        if stable and recommended is None:
            recommended = {
                "popsize": int(popsize),
                "maxiter": int(maxiter),
                "total_spread": float(spread),
                "chi2_data_mean": float(np.nanmean(chi2s)),
                "stable": True,
            }

    if recommended is None:
        last = stages[-1]
        recommended = {
            "popsize": int(last["popsize"]),
            "maxiter": int(last["maxiter"]),
            "total_spread": float(last["total_spread"]),
            "chi2_data_mean": float(last["chi2_data_mean"]),
            "stable": False,
        }
        warning = (
            f"No DE budget met spread tol={spread_tol:g}; "
            f"recommending last stage popsize={recommended['popsize']}, "
            f"maxiter={recommended['maxiter']} (spread={recommended['total_spread']:.3g})."
        )

    return {
        "stages": stages,
        "recommended": recommended,
        "spread_tol": float(spread_tol),
        "n_seeds": int(n_seeds),
        "warning": warning,
    }


def lcurve_corner_index(
    data_misfit: Sequence[float],
    reg_norm: Sequence[float],
) -> int:
    """Index of maximum discrete curvature on the log-log L-curve.

    Points are used in the order given (caller should sort by increasing lambda).
    Drops non-finite or non-positive coordinates.
    """
    d = np.asarray(data_misfit, dtype=float)
    r = np.asarray(reg_norm, dtype=float)
    if d.size != r.size or d.size < 3:
        return int(np.nanargmin(d)) if d.size else 0

    valid = np.isfinite(d) & np.isfinite(r) & (d > 0.0) & (r > 0.0)
    idx_map = np.where(valid)[0]
    if idx_map.size < 3:
        # Fall back: smallest data misfit among valid, else 0.
        if idx_map.size:
            local = int(np.argmin(d[idx_map]))
            return int(idx_map[local])
        return 0

    x = np.log10(d[idx_map])
    y = np.log10(r[idx_map])
    # Normalize to unit box so curvature is scale-invariant.
    x = (x - np.min(x)) / max(float(np.max(x) - np.min(x)), 1e-300)
    y = (y - np.min(y)) / max(float(np.max(y) - np.min(y)), 1e-300)

    curv = np.full(idx_map.size, -np.inf, dtype=float)
    for i in range(1, idx_map.size - 1):
        x1, y1 = x[i - 1], y[i - 1]
        x2, y2 = x[i], y[i]
        x3, y3 = x[i + 1], y[i + 1]
        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x3 - x2, y3 - y2
        # Menger curvature of three points (area / product of sides).
        area = abs(dx1 * dy2 - dy1 * dx2)
        s1 = float(np.hypot(dx1, dy1))
        s2 = float(np.hypot(dx2, dy2))
        s3 = float(np.hypot(x3 - x1, y3 - y1))
        denom = s1 * s2 * s3
        if denom <= 0.0:
            continue
        curv[i] = area / denom

    local = int(np.argmax(curv))
    return int(idx_map[local])


def tune_lambda_lcurve(
    cfg: Mapping[str, Any],
    tx_entry: Mapping[str, Any],
    *,
    lambdas: Sequence[float] = DEFAULT_LAMBDA_GRID,
    seed: Optional[int] = None,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Sweep lambda at fixed DE popsize/maxiter; pick L-curve corner."""
    if "calibration" not in cfg:
        raise ValueError("cfg must include calibration from notebook 02.")
    seed = int(cfg.get("seed", 42) if seed is None else seed)
    lam_list = [float(v) for v in lambdas]
    jobs = [(cfg, tx_entry, lam, seed) for lam in lam_list]
    runs = _parallel_map(_run_de_with_lambda, jobs, n_jobs=int(n_jobs))
    # Ensure ascending lambda order for curvature.
    order = np.argsort(lam_list)
    runs_sorted = [runs[i] for i in order]
    lams_sorted = [lam_list[i] for i in order]
    data = [float(r["data_misfit"]) for r in runs_sorted]
    regs = [float(r["reg_norm"]) for r in runs_sorted]
    corner_i = lcurve_corner_index(data, regs)
    recommended_lambda = float(lams_sorted[corner_i])

    return {
        "lambdas": lams_sorted,
        "runs": runs_sorted,
        "data_misfit": data,
        "reg_norm": regs,
        "corner_index": int(corner_i),
        "recommended": {
            "reg_lambda": recommended_lambda,
            "data_misfit": float(data[corner_i]),
            "reg_norm": float(regs[corner_i]),
            "chi2_data": float(runs_sorted[corner_i]["chi2_data"]),
            "total": float(runs_sorted[corner_i]["total"]),
        },
        "seed": int(seed),
        "popsize": int(cfg["popsize"]),
        "maxiter": int(cfg["maxiter"]),
    }


__all__ = [
    "DEFAULT_DE_BUDGETS",
    "DEFAULT_LAMBDA_GRID",
    "DEFAULT_N_TUNE_SEEDS",
    "DEFAULT_SEED_SPREAD_TOL",
    "build_bounds",
    "lcurve_corner_index",
    "relative_objective_spread",
    "run_de_once",
    "split_objective",
    "tune_de_budget",
    "tune_lambda_lcurve",
    "unpack_model_params",
]
