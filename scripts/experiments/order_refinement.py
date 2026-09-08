"""Separate a CONSISTENCY error from a DISCRETISATION error by refining dx.

Task-1's claim is that rock-em's tabulated (dispersion-optimised, Holberg-type)
staggered first-derivative coefficients violate the consistency condition
``sum_n c_n (2n+1) = 1`` - by -2.299 % at order 2 - so every first derivative is
under-scaled by that factor REGARDLESS of dx. The decisive signature of such an
error is that **it does not converge away under grid refinement**, whereas an
ordinary truncation/interpolation error falls like dx^2.

This script measures the FDTD-vs-analytic scale ``C(f)`` on the same Earth model
at two grid spacings for two stencil orders, and reports how ``|C|/dx^2`` moves.

Expected, if the hypothesis holds:
  order 2:  |C|/dx^2 stays put when dx halves   (non-converging => consistency)
  order 6:  the (much smaller) residual falls ~4x  (converging => discretisation)

``dx`` is steered through ``cells_per_min_offset`` because that is the binding
constraint at this survey's 13.1 m minimum offset: dx = min(induction limit,
min_offset / cells_per_min_offset).

Usage:
    python scripts/experiments/order_refinement.py --orders 2 6 --cells-per-offset 8 32
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.fdtd_analytic_calibration import (  # noqa: E402
    METHOD_HOMOGENEOUS, METHOD_LATERAL_AVERAGE,
)
from scripts.modules.headless import (  # noqa: E402
    SetupParams, build_forward_inputs, run_calibration,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 6])
    ap.add_argument("--cells-per-offset", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--methods", nargs="+",
                    default=[METHOD_LATERAL_AVERAGE, METHOD_HOMOGENEOUS])
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--workspace", default="workspace/2D/refinement")
    ap.add_argument("--out", default="workspace/2D/refinement/refinement.json")
    args = ap.parse_args()

    results = {}
    for order in args.orders:
        for cpo in args.cells_per_offset:
            run_dir = Path(args.workspace) / f"o{order}_c{cpo}"
            p = replace(SetupParams(), fd_order=order, cells_per_min_offset=cpo)
            meta = build_forward_inputs(run_dir, p)
            for method in args.methods:
                cal = run_calibration(run_dir, method=method, nproc=args.nproc, verbose=True)
                c = np.asarray(cal["C_hxhz_shared"], dtype=complex)
                dx2 = float(cal["dx_squared"])
                results[f"order{order}_dx{meta['dx_model_target_m']:g}_{method}"] = {
                    "order": order, "dx_m": meta["dx_model_target_m"],
                    "nt": meta["nt_model"], "method": method,
                    "freqs_hz": list(cal["freqs_hz"]),
                    "C_over_dx2": (np.abs(c) / dx2).tolist(),
                    "phase_deg": np.angle(c, deg=True).tolist(),
                    "scatter_hx_pct": cal["scatter_hx_pct"],
                    "scatter_hz_pct": cal["scatter_hz_pct"],
                    "wall_s": cal["fdtd_wall_s"],
                }
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(results, indent=2) + "\n")

    print("\n=== summary: mean deficit (1 - |C|/dx^2) in %, and its spread over the band ===")
    print(f"{'key':>52} {'mean %':>9} {'spread %':>9}")
    for k, v in results.items():
        arr = np.asarray(v["C_over_dx2"])
        print(f"{k:>52} {100*(1-arr.mean()):9.4f} {100*(arr.max()-arr.min()):9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
