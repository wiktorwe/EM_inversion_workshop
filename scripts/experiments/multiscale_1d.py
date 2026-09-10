"""Task 3, prototyped in 1D: a multi-scale frequency ladder vs one joint inversion.

The task's instruction is to settle the ORDERING, the per-stage budget and the
regularisation schedule here - where one inversion costs seconds and hundreds of
trials are affordable - before spending 2D FWI time on it.

Why a frequency ladder is the right structure for diffusive EM, stated honestly:

* **Skin depth is the scale.** delta = sqrt(2 rho / (omega mu)) ~ 1/sqrt(f), so
  the frequency ladder is directly a ladder in depth of investigation and in the
  spatial wavelength of the sensitivity kernel. At the workshop's mean 28 Ohm-m
  the skin depths are 84.5 / 59.7 / 42.2 / 34.5 m at 1 / 2 / 4 / 6 kHz. Low
  frequency senses deep and smooth, high frequency shallow and sharp. Building
  the long-wavelength structure first is worth doing because the high-frequency
  data cannot constrain it, and if it is wrong every later stage inherits it.
* **Cost.** Low frequency runs on a coarse grid (Task 2: 1.60 m at 1 kHz against
  0.80 m at 6 kHz), so the early stages that do the most model-changing work are
  also the cheapest per iteration.
* **NOT cycle skipping.** The diffusive field accumulates roughly r/delta radians
  of phase, so a phase-based misfit can genuinely wrap once offsets reach several
  skin depths - but this survey sits at 0.15-0.73 skin depths, where it cannot.
  The honest reasons multi-scale helps HERE are conditioning and cost. The
  cycle-skipping argument only becomes real at true UDAR offsets.

Judged on TWO criteria, because multi-scale is not automatically better:

  1. data misfit evaluated on ALL frequencies, not just the last stage's - a
     ladder that fits 6 kHz beautifully while drifting off the 1 kHz data has
     failed;
  2. model error against the KNOWN truth (`examples/Fault_1.sgy`) - this is a
     synthetic study, so model error can be measured rather than inferred from
     misfit. This is the more important criterion here.

Both are reported after EVERY stage, not just at the end: if the benefit is
concentrated in one stage, or a stage makes things worse, that is the useful
finding and it should be visible rather than averaged away.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.analytic_1d_forward import announce_quadrature_policy  # noqa: E402
from scripts.modules.fd_visualization import compute_gains_for_fd_outputs  # noqa: E402
from scripts.modules.fdtd_analytic_calibration import (  # noqa: E402
    calibration_for_inversion, load_setup_metadata,
)
from scripts.modules.inversion_1d import (  # noqa: E402
    build_bounds, complex_gain_objective, unpack_model_params,
)
from scripts.modules.multiscale_2d import read_sg_grid  # noqa: E402


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_tx_features(fwd_dir: Path) -> tuple[dict, dict]:
    """Per-Tx complex channel gains, exactly as notebook 05's `extract_features`."""
    fwd_dir = Path(fwd_dir)
    meta = load_setup_metadata(fwd_dir / "setup_metadata.json")
    freqs = np.asarray(meta["flist_hz"], dtype=float)
    g = compute_gains_for_fd_outputs(
        fwd_dir / "Data" / "Hxshot.rss", fwd_dir / "Data" / "Hzshot.rss",
        fwd_dir / "wav2d.rss", freqs=freqs,
        f_min_hz=float(meta["f_min_hz"]),
        n_periods_extract=float(meta["n_periods_extract"]),
    )
    geo = g["geometry"]
    tx_idx = np.asarray(geo["tx_idx_per_trace"], dtype=int)
    tx_unique = np.asarray(geo["tx_unique"], dtype=float)
    rx_x = np.asarray(geo["rx_x"], dtype=float)
    rx_z = np.asarray(geo["rx_z"], dtype=float)
    tx_data = {}
    for t in np.unique(tx_idx):
        tr = np.where(tx_idx == int(t))[0]
        tx_data[int(t)] = {
            "tx_id": int(t),
            "tx_x": float(tx_unique[int(t), 0]),
            "tx_z": float(tx_unique[int(t), 1]),
            "off_x": rx_x[tr] - float(tx_unique[int(t), 0]),
            "off_z": rx_z[tr] - float(tx_unique[int(t), 1]),
            "freqs": freqs,
            "obs_hx_gain": np.asarray(g["Hx"]["gain"][:, tr], dtype=complex),
            "obs_hz_gain": np.asarray(g["Hz"]["gain"][:, tr], dtype=complex),
        }
    return meta, tx_data


def true_profile(fwd_dir: Path, tx_x: float) -> tuple[np.ndarray, np.ndarray]:
    """(z, rho) of the TRUE model beneath one transmitter, from production sg.rss.

    Uses `multiscale_2d.read_sg_grid` rather than a third local copy of the
    axis convention - samples sit at `o + k*d`, not at cell centres, and getting
    that wrong shifts the truth by half a cell (0.4 m here), which is enough to
    change |Hz| by ~5 % at 6 kHz.
    """
    g = read_sg_grid(Path(fwd_dir) / "sg.rss")
    j = int(np.argmin(np.abs(g["x"] - tx_x)))
    return g["z"], 1.0 / np.clip(g["sigma"][j, :], 1e-12, None)


def model_error(params, cfg, z_true, rho_true) -> float:
    """RMS log10-resistivity error against the true model over the depth window."""
    rho, _thk, depth_rel = unpack_model_params(
        params, cfg["n_layers"], cfg["z_start_rel"], cfg["z_end_rel"])
    tx_z = cfg["tx_z"]
    edges = np.concatenate([[tx_z + cfg["z_start_rel"]], tx_z + np.asarray(depth_rel),
                            [tx_z + cfg["z_end_rel"]]])
    m = (z_true >= edges[0]) & (z_true <= edges[-1])
    zz, rt = z_true[m], rho_true[m]
    if zz.size == 0:
        return float("nan")
    idx = np.clip(np.searchsorted(edges[1:-1], zz, side="right"), 0, rho.size - 1)
    return float(np.sqrt(np.mean((np.log10(rho[idx]) - np.log10(rt)) ** 2)))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def _seeded_population(x0, bounds, popsize, rng, radius_frac):
    """DE start population centred on the previous stage's model.

    scipy's `differential_evolution` takes `init` as an (S, N) array. Stage k+1
    starts from stage k's answer plus shrinking Gaussian scatter, so the ladder
    actually inherits the previous model instead of re-searching the whole prior.
    Member 0 is the previous best EXACTLY, so a stage can never do worse than its
    own starting point by more than the optimiser's own noise.
    """
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    n = lo.size
    size = max(5, int(popsize) * n)
    pop = rng.normal(0.0, radius_frac, size=(size, n)) * (hi - lo) + np.asarray(x0, dtype=float)
    pop[0] = np.asarray(x0, dtype=float)
    return np.clip(pop, lo, hi)


def run_stage(tx_entry, cfg, freq_mask, *, reg_lambda, seed, maxiter, popsize,
              x0=None, radius_frac=0.12):
    bounds = build_bounds(cfg["n_layers"], cfg["log10_rho_min"], cfg["log10_rho_max"],
                          cfg["log10_thk_min"], cfg["log10_thk_max"])
    cal = cfg["calibration"]
    obj = lambda p: complex_gain_objective(
        p, tx_entry=tx_entry, n_layers=cfg["n_layers"],
        z_start_rel=cfg["z_start_rel"], z_end_rel=cfg["z_end_rel"], eps_r=cfg["eps_r"],
        reg_lambda=reg_lambda, w_cxx=cfg["w_Cxx"], w_cxz=cfg["w_Cxz"],
        sigma_hx=cal["sigma_hx"], sigma_hz=cal["sigma_hz"], C=cal["C"], freq_mask=freq_mask,
    )
    rng = np.random.default_rng(seed)
    init = "latinhypercube" if x0 is None else _seeded_population(
        x0, bounds, popsize, rng, radius_frac)
    out = differential_evolution(obj, bounds=bounds, maxiter=int(maxiter),
                                 popsize=int(popsize), seed=int(seed), polish=False,
                                 workers=1, updating="deferred", init=init)
    return np.asarray(out.x, dtype=float), float(out.fun)


def full_band_chi2(params, tx_entry, cfg) -> float:
    """Reduced misfit on ALL frequencies, unregularised - the honest scoreboard.

    A ladder must be judged on the whole band, not on the stage it happens to
    have finished on.
    """
    cal = cfg["calibration"]
    nfreq = np.asarray(tx_entry["freqs"]).size
    mis = complex_gain_objective(
        params, tx_entry=tx_entry, n_layers=cfg["n_layers"],
        z_start_rel=cfg["z_start_rel"], z_end_rel=cfg["z_end_rel"], eps_r=cfg["eps_r"],
        reg_lambda=0.0, w_cxx=cfg["w_Cxx"], w_cxz=cfg["w_Cxz"],
        sigma_hx=cal["sigma_hx"], sigma_hz=cal["sigma_hz"], C=cal["C"], freq_mask=None,
    )
    nrx = np.asarray(tx_entry["off_x"]).size
    n_data = 4 * nfreq * nrx      # complex Hx + complex Hz
    return float(mis) / max(n_data, 1)


def invert_tx(tx_entry, cfg, strategy: str, *, seed: int, schedule: dict) -> dict:
    freqs = np.asarray(tx_entry["freqs"], dtype=float)
    nfreq = freqs.size
    z_true, rho_true = cfg["true_profile"]
    trajectory = []
    t0 = time.perf_counter()

    if strategy == "single":
        mask = np.ones(nfreq, dtype=bool)
        x, _ = run_stage(tx_entry, cfg, mask, reg_lambda=schedule["reg_final"],
                         seed=seed, maxiter=schedule["maxiter_single"],
                         popsize=schedule["popsize"])
        trajectory.append({"stage": "all", "freqs": freqs.tolist(),
                           "reg_lambda": schedule["reg_final"],
                           "chi2_all": full_band_chi2(x, tx_entry, cfg),
                           "model_err": model_error(x, cfg, z_true, rho_true)})
    elif strategy in ("ladder", "ladder_joint", "cumulative"):
        order = np.argsort(freqs)          # lowest first
        x = None
        for k, i in enumerate(order):
            if strategy == "cumulative":
                mask = np.zeros(nfreq, dtype=bool); mask[order[: k + 1]] = True
            else:
                mask = np.zeros(nfreq, dtype=bool); mask[i] = True
            # regularisation relaxes as frequency rises: heavy while the model
            # is long-wavelength, light once the high frequencies are asked to
            # add detail. Without this the ladder converges smooth and the later
            # stages cannot add what they exist to add.
            reg = schedule["reg_final"] * (freqs.max() / freqs[i]) ** schedule["reg_exponent"]
            x, _ = run_stage(tx_entry, cfg, mask, reg_lambda=reg, seed=seed + k,
                             maxiter=schedule["maxiter_stage"], popsize=schedule["popsize"],
                             x0=x, radius_frac=schedule["radius_frac"])
            trajectory.append({"stage": f"f={freqs[i]:.0f}", "freqs": freqs[mask].tolist(),
                               "reg_lambda": reg,
                               "chi2_all": full_band_chi2(x, tx_entry, cfg),
                               "model_err": model_error(x, cfg, z_true, rho_true)})
        if strategy == "ladder_joint":
            mask = np.ones(nfreq, dtype=bool)
            x, _ = run_stage(tx_entry, cfg, mask, reg_lambda=schedule["reg_final"],
                             seed=seed + 100, maxiter=schedule["maxiter_joint"],
                             popsize=schedule["popsize"], x0=x,
                             radius_frac=schedule["radius_frac_joint"])
            trajectory.append({"stage": "joint-all", "freqs": freqs.tolist(),
                               "reg_lambda": schedule["reg_final"],
                               "chi2_all": full_band_chi2(x, tx_entry, cfg),
                               "model_err": model_error(x, cfg, z_true, rho_true)})
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    return {"strategy": strategy, "params": x.tolist(), "trajectory": trajectory,
            "chi2_all": trajectory[-1]["chi2_all"],
            "model_err": trajectory[-1]["model_err"],
            "wall_s": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _one(tx_id, tx_entry, cfg, strategy, seed, schedule):
    r = invert_tx(tx_entry, cfg, strategy, seed=seed, schedule=schedule)
    r.update({"tx_id": tx_id, "tx_x": float(tx_entry["tx_x"]), "seed": seed})
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-dir", default="workspace/2D/forward")
    ap.add_argument("--tx", type=int, nargs="+", default=None,
                    help="transmitter indices (default: all)")
    ap.add_argument("--strategies", nargs="+",
                    default=["single", "ladder", "ladder_joint", "cumulative"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-layers", type=int, default=5)
    ap.add_argument("--z-start-rel", type=float, default=-60.0)
    ap.add_argument("--z-end-rel", type=float, default=60.0)
    ap.add_argument("--reg-final", type=float, default=500.0)
    ap.add_argument("--reg-exponent", type=float, default=1.0,
                    help="reg_lambda = reg_final * (f_max/f_stage)**exponent; 0 disables the schedule")
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--maxiter-single", type=int, default=120)
    ap.add_argument("--maxiter-stage", type=int, default=30)
    ap.add_argument("--maxiter-joint", type=int, default=30)
    ap.add_argument("--radius-frac", type=float, default=0.12)
    ap.add_argument("--radius-frac-joint", type=float, default=0.06)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default="workspace/1D/multiscale/multiscale_1d.json")
    args = ap.parse_args()

    announce_quadrature_policy()
    fwd = Path(args.forward_dir)
    meta, tx_data = load_tx_features(fwd)
    cal = calibration_for_inversion(fwd / "setup_metadata.json")
    tx_ids = sorted(tx_data) if args.tx is None else list(args.tx)

    schedule = {
        "reg_final": args.reg_final, "reg_exponent": args.reg_exponent,
        "popsize": args.popsize, "maxiter_single": args.maxiter_single,
        "maxiter_stage": args.maxiter_stage, "maxiter_joint": args.maxiter_joint,
        "radius_frac": args.radius_frac, "radius_frac_joint": args.radius_frac_joint,
    }
    print(f"schedule: {schedule}")
    print(f"calibration: method={cal['method']}, C={np.abs(cal['C'])}")
    print(f"{len(tx_ids)} transmitters, {len(args.seeds)} seed(s), strategies={args.strategies}")

    # Jobs are built as fully-resolved (tx_entry, cfg) pairs and dispatched to a
    # MODULE-LEVEL worker: joblib's default backend pickles the callable, and a
    # nested closure is not picklable, so a nested `one()` would work at
    # n_jobs=1 and fail only under parallelism.
    jobs = []
    for t in tx_ids:
        tx_entry = tx_data[t]
        cfg = {
            "n_layers": args.n_layers, "z_start_rel": args.z_start_rel,
            "z_end_rel": args.z_end_rel, "eps_r": float(meta["eps_r_used"]),
            "log10_rho_min": np.log10(float(meta["rho_min_ohm_m"])),
            "log10_rho_max": np.log10(float(meta["rho_max_ohm_m"])),
            "log10_thk_min": np.log10(5.0), "log10_thk_max": np.log10(50.0),
            "w_Cxx": 1.0, "w_Cxz": 1.0, "calibration": cal,
            "tx_z": float(tx_entry["tx_z"]),
            "true_profile": true_profile(fwd, float(tx_entry["tx_x"])),
        }
        for strat in args.strategies:
            for sd in args.seeds:
                jobs.append((t, tx_entry, cfg, strat, sd, schedule))

    if args.n_jobs != 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(_one)(*j) for j in jobs)
    else:
        results = []
        for j in jobs:
            results.append(_one(*j))
            print(f"  tx {j[0]:2d} {j[3]:>13} seed {j[4]}: "
                  f"chi2_all={results[-1]['chi2_all']:9.3f} "
                  f"model_err={results[-1]['model_err']:.4f} "
                  f"({results[-1]['wall_s']:.1f} s)", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schedule": schedule, "results": results,
                               "forward_dir": str(fwd)}, indent=2) + "\n")

    print("\n=== head-to-head, averaged over transmitters and seeds ===")
    print(f"{'strategy':>14} {'chi2 (all freqs)':>17} {'model err (log10)':>18} {'wall s':>8}")
    for s in args.strategies:
        rs = [r for r in results if r["strategy"] == s]
        if not rs:
            continue
        print(f"{s:>14} {np.mean([r['chi2_all'] for r in rs]):17.3f} "
              f"{np.mean([r['model_err'] for r in rs]):18.4f} "
              f"{np.mean([r['wall_s'] for r in rs]):8.1f}")

    print("\n=== stage-by-stage trajectory (mean over tx/seeds) ===")
    for s in args.strategies:
        rs = [r for r in results if r["strategy"] == s]
        if not rs:
            continue
        print(f"  {s}")
        for k in range(len(rs[0]["trajectory"])):
            name = rs[0]["trajectory"][k]["stage"]
            print(f"    after {name:>10}: chi2_all={np.mean([r['trajectory'][k]['chi2_all'] for r in rs]):9.3f} "
                  f"model_err={np.mean([r['trajectory'][k]['model_err'] for r in rs]):.4f} "
                  f"(lambda={rs[0]['trajectory'][k]['reg_lambda']:.4g})")

    print("\n=== per-transmitter model error (the fault sits at x = 1478 m) ===")
    print(f"{'tx':>4} {'x [m]':>8} " + " ".join(f"{s:>14}" for s in args.strategies))
    for t in tx_ids:
        row = []
        for s in args.strategies:
            rs = [r for r in results if r["strategy"] == s and r["tx_id"] == t]
            row.append(f"{np.mean([r['model_err'] for r in rs]):14.4f}" if rs else f"{'':>14}")
        xt = next(r["tx_x"] for r in results if r["tx_id"] == t)
        print(f"{t:4d} {xt:8.1f} " + " ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
