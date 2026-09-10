"""Does the analytic error budget replace interface snapping, or need it?

Snapping makes the candidate model mimic each dataset's interface quantisation.
The alternative is to admit the grid cannot resolve below half a cell and put
that in `sigma`, so the inversion never chases the detail in the first place.
Doing BOTH double-counts: the candidate is snapped (removing the error) while
sigma is inflated for it.

So this decides, by running the same inversion three ways and comparing what
matters - the RECOVERED MODEL against the truth, not chi-squared. Chi-squared is
not comparable across these: sigma differs, so it moves for reasons that have
nothing to do with the model being better.

    python scripts/experiments/sigma_budget_ab.py --n-tx 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.headless import matrix_setup  # noqa: E402
from scripts.modules.inversion_1d import (  # noqa: E402
    analytic_tensor_calibration, build_bounds, load_tensor_features,
    n_tensor_data, tensor_calibration, tensor_objective_parts,
    unpack_model_params,
)
from scripts.modules.segy import read_resistivity_from_segy  # noqa: E402
from scripts.modules.inversion_1d import blocky_layers_from_trace  # noqa: E402


def truth_log_rho(segy, tx_x, tx_z, zs, ze, zg):
    seg = read_resistivity_from_segy(str(segy))
    z = np.asarray(seg["z"], float)
    ix = int(np.argmin(np.abs(np.asarray(seg["x"], float) - tx_x)))
    d_abs, res = blocky_layers_from_trace(np.asarray(seg["resistivity"], float)[:, ix],
                                          z0=float(z[0]), dz=float(z[1] - z[0]))
    idx = np.searchsorted(np.asarray(d_abs, float), zg + tx_z)
    return np.log10(np.asarray(res, float)[np.clip(idx, 0, len(res) - 1)])


def model_log_rho(params, n_layers, zs, ze, zg):
    rho, _thk, depth = unpack_model_params(params, n_layers, zs, ze)
    idx = np.searchsorted(depth, zg)
    return np.log10(rho[np.clip(idx, 0, rho.size - 1)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-dir", default="workspace/2D/forward")
    ap.add_argument("--segy", default="examples/Fault_1.sgy")
    ap.add_argument("--n-tx", type=int, default=3)
    ap.add_argument("--n-layers", type=int, default=5)
    ap.add_argument("--z-start", type=float, default=-60.0)
    ap.add_argument("--z-end", type=float, default=60.0)
    ap.add_argument("--maxiter", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reg-lambda", type=float, default=500.0)
    ap.add_argument("--components", default="",
                    help="Comma-separated subset, e.g. Cxx,Czz. Default: all present.")
    ap.add_argument("--out", default="workspace/2D/sigma_budget_ab.json")
    args = ap.parse_args()

    m = matrix_setup(Path(args.forward_dir))
    feats = load_tensor_features({s: v for s, v in m["by_source"].items()})
    comps = tuple(c.strip() for c in args.components.split(",") if c.strip()) \
        or tuple(feats["components"])
    meta = json.loads(Path(m["representative"]).read_text())
    order = int(meta["fd_order"])
    zs, ze = args.z_start, args.z_end
    zg = np.linspace(zs, ze, 400)

    base = {"n_layers": args.n_layers, "z_start_rel": zs, "z_end_rel": ze,
            "eps_r": m["eps_r"], "dx": m["dx"], "fd_order": order,
            "reg_lambda": args.reg_lambda, "components": comps,
            "log10_rho_min": 0.0, "log10_rho_max": np.log10(150.0),
            "log10_thk_min": np.log10(5.0), "log10_thk_max": np.log10(50.0)}
    fitted = tensor_calibration({s: [str(p) for p in v]
                                 for s, v in m["meta_paths"].items()})
    snap = (m["dx"], m["snap_origin_m"])

    modes = [
        ("fitted sigma + snapping",  "fitted",   True),
        ("budget sigma + snapping",  "analytic", True),
        ("budget sigma, NO snapping", "analytic", False),
    ]
    bounds = build_bounds(args.n_layers, base["log10_rho_min"], base["log10_rho_max"],
                          base["log10_thk_min"], base["log10_thk_max"])
    tx_ids = sorted(feats["tx_data"])[: max(1, args.n_tx)]
    report = {"modes": {}, "tx_ids": [int(t) for t in tx_ids]}

    print(f"{args.n_layers} layers, DE popsize={args.popsize} maxiter={args.maxiter}, "
          f"lambda={args.reg_lambda:g}, {len(tx_ids)} Tx, components {', '.join(comps)}\n")
    print(f"{'configuration':>26} {'model err (rms log10 rho)':>27} {'data chi2':>11}")
    for label, cal_kind, use_snap in modes:
        errs, chis = [], []
        for tx_id in tx_ids:
            tx = feats["tx_data"][tx_id]
            # Mirror the notebook: snapping lives in cfg, and the analytic
            # calibration reads it to suppress its own quantisation term (the
            # two are alternatives, not partners). Passing snapping only to the
            # objective would leave the budget double-counting.
            kw = dict(snap_dz=snap[0], snap_origin_m=snap[1]) if use_snap else {}
            cfg_i = dict(base, **kw)
            cal = (analytic_tensor_calibration(cfg_i, tx, components=comps)
                   if cal_kind == "analytic" else fitted)

            def obj(p):
                return tensor_objective_parts(
                    p, tx, args.n_layers, zs, ze, m["eps_r"], args.reg_lambda, cal,
                    components=comps, **kw)[2]

            out = differential_evolution(obj, bounds=bounds, maxiter=args.maxiter,
                                         popsize=args.popsize, seed=args.seed,
                                         polish=False, workers=1, updating="deferred")
            best = np.asarray(out.x, float)
            data_mis, _reg, _tot = tensor_objective_parts(
                best, tx, args.n_layers, zs, ze, m["eps_r"], args.reg_lambda, cal,
                components=comps, **kw)
            errs.append(float(np.sqrt(np.mean(
                (model_log_rho(best, args.n_layers, zs, ze, zg)
                 - truth_log_rho(args.segy, float(tx["tx_x"]), float(tx["tx_z"]),
                                 zs, ze, zg)) ** 2))))
            chis.append(data_mis / max(n_tensor_data(tx, comps), 1))
        report["modes"][label] = {"model_error": errs, "chi2_data": chis,
                                  "model_error_mean": float(np.mean(errs)),
                                  "chi2_data_mean": float(np.mean(chis))}
        print(f"{label:>26} {np.mean(errs):>27.4f} {np.mean(chis):>11.4f}")

    best_label = min(report["modes"], key=lambda k: report["modes"][k]["model_error_mean"])
    print(f"\nbest recovered model: {best_label}")
    print("NOTE: chi2 is NOT comparable across rows - sigma differs between them.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
