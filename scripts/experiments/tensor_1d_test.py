"""Does the 1D inversion actually work with all four tensor components?

Two checks, in the order that matters:

1. CONSISTENCY - evaluate the KNOWN true model against the real FDTD data for
   every component. If the truth does not fit, nothing downstream means
   anything, and no amount of optimiser tuning will fix it. This is the check
   that exposed a 21.6 degree phase error the calibration could not see.

2. RECOVERY - actually invert, from a uniform start, and see whether the model
   comes back. Reported for the Kx pair alone and for the full tensor, so the
   effect of adding Czx/Czz is visible rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from scripts.modules.inversion_1d import (  # noqa: E402
    TENSOR_COMPONENTS, build_bounds, forward_tensor_for_tx, load_tensor_features,
    n_tensor_data, tensor_calibration, tensor_objective, unpack_model_params,
)
from scripts.modules.multiscale_2d import read_sg_grid  # noqa: E402

ALL = ("Cxx", "Cxz", "Czx", "Czz")


def true_params(fwd_dir, tx_x, tx_z, zs, ze):
    """The true 5-layer vector beneath one transmitter, from production sg.rss."""
    g = read_sg_grid(Path(fwd_dir) / "sg.rss")
    j = int(np.argmin(np.abs(g["x"] - tx_x)))
    z, rho = g["z"], 1.0 / np.clip(g["sigma"][j, :], 1e-12, None)
    lr = np.log10(rho)
    k = np.where(np.abs(np.diff(lr)) > 1e-6)[0]
    faces = 0.5 * (z[k] + z[k + 1]) - tx_z          # interfaces, tx-relative
    faces = faces[(faces > zs) & (faces < ze)]
    edges = np.concatenate([[zs], faces, [ze]])
    thk = np.diff(edges)
    rhos = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (z - tx_z >= a) & (z - tx_z < b)
        rhos.append(float(np.exp(np.mean(np.log(rho[m])))) if m.any() else 25.0)
    rhos.append(rhos[-1])                            # halfspace below z_end
    return np.concatenate([np.log10(rhos), np.log10(thk)]), len(rhos), (z, rho)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hx-dir", default="workspace/2D/forward")
    ap.add_argument("--hz-dir", default="workspace/2D/forward_hz")
    ap.add_argument("--meta", default="workspace/2D/forward/setup_metadata.json")
    ap.add_argument("--tx", type=int, default=0)
    ap.add_argument("--z-start", type=float, default=-60.0)
    ap.add_argument("--z-end", type=float, default=60.0)
    ap.add_argument("--maxiter", type=int, default=120)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feats = load_tensor_features({"HX": args.hx_dir, "HZ": args.hz_dir})
    cal = tensor_calibration({"HX": args.meta, "HZ": args.meta})
    meta = json.loads(Path(args.meta).read_text())
    eps_r = float(meta["eps_r_used"])
    e = feats["tx_data"][args.tx]
    zs, ze = args.z_start, args.z_end

    print(f"components loaded : {feats['components']}")
    print(f"sources           : {feats['sources']}")
    print(f"|C| per source    : "
          + ", ".join(f"{s}={np.abs(cal['C'][s])[0]:.5f}" for s in sorted(cal["C"])))
    print(f"sigma per component: "
          + ", ".join(f"{c}={cal['sigma'][c][0]:.3e}" for c in ALL))

    # ---- 1. consistency: does the TRUE model fit every component? ----------
    p_true, n_layers, (z, rho) = true_params(args.hx_dir, e["tx_x"], e["tx_z"], zs, ze)
    pred = forward_tensor_for_tx(p_true, e, n_layers, zs, ze, eps_r, components=ALL)
    print(f"\n=== 1. TRUE model vs FDTD data, tx {args.tx} at x={e['tx_x']:.1f} m "
          f"({n_layers} layers) ===")
    print(f"{'comp':>6} {'amp ratio':>10} {'phase':>9} {'resid/sigma':>12}")
    for c in ALL:
        src = TENSOR_COMPONENTS[c][0]
        a = np.asarray(cal["C"][src])[:, None] * pred[c]
        b = np.asarray(e["obs"][c])
        s = np.asarray(cal["sigma"][c])[:, None]
        print(f"{c:>6} {np.max(np.abs(a)/np.abs(b)):10.5f} "
              f"{np.max(np.abs(np.angle(a/b, deg=True))):+8.3f}° "
              f"{np.max(np.abs(a-b)/s):12.3f}")
    chi2_true = tensor_objective(p_true, e, n_layers, zs, ze, eps_r, 0.0, cal,
                                 components=ALL) / max(n_tensor_data(e, ALL), 1)
    print(f"  reduced chi-squared of the TRUE model, all four components: {chi2_true:.4f}")
    print("  (<1 means the truth fits inside the assumed noise - the precondition "
          "for any\n   inversion result below to be meaningful)")

    # ---- 2. recovery: invert from a uniform start -------------------------
    lo, hi = np.log10(float(meta["rho_min_ohm_m"])), np.log10(float(meta["rho_max_ohm_m"]))
    bounds = build_bounds(n_layers, lo, hi, np.log10(5.0), np.log10(50.0))

    def model_err(p):
        r, _t, d = unpack_model_params(p, n_layers, zs, ze)
        edges = np.concatenate([[e["tx_z"] + zs], e["tx_z"] + np.asarray(d),
                                [e["tx_z"] + ze]])
        m = (z >= edges[0]) & (z <= edges[-1])
        idx = np.clip(np.searchsorted(edges[1:-1], z[m], side="right"), 0, r.size - 1)
        return float(np.sqrt(np.mean((np.log10(r[idx]) - np.log10(rho[m])) ** 2)))

    print(f"\n=== 2. recovery from a uniform start (DE, maxiter={args.maxiter}) ===")
    print(f"{'components':>26} {'chi2':>10} {'model err':>10}")
    out = {}
    for comps in (("Cxx", "Cxz"), ALL):
        obj = lambda p, c=comps: tensor_objective(
            p, e, n_layers, zs, ze, eps_r, 500.0, cal, components=c)
        r = differential_evolution(obj, bounds=bounds, maxiter=args.maxiter,
                                   popsize=args.popsize, seed=args.seed, polish=False,
                                   workers=1, updating="deferred")
        # score every run on the SAME yardstick: all four components, unregularised
        chi2 = tensor_objective(r.x, e, n_layers, zs, ze, eps_r, 0.0, cal,
                                components=ALL) / max(n_tensor_data(e, ALL), 1)
        out["+".join(comps)] = {"chi2_all": float(chi2), "model_err": model_err(r.x)}
        print(f"{'+'.join(comps):>26} {chi2:10.4f} {model_err(r.x):10.4f}")
    print(f"{'TRUE model, for reference':>26} {chi2_true:10.4f} {model_err(p_true):10.4f}")
    print("\nBoth rows are scored on all four components, so adding Czx/Czz to the fit")
    print("is compared on the same yardstick as leaving them out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
