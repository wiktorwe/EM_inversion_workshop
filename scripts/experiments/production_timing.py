"""Measured wall-clock cost of the order-2 -> order-6 stencil change.

Task 1 asks for a REAL number, not an estimate, so this times one full
production forward run (all 30 transmitters on `examples/Fault_1.sgy`, the same
survey and the same MPI launch the workshop's own `runmod.sh` uses) at both
stencil orders, back to back on an otherwise idle machine.

The order-6 run is written to `workspace/2D/forward`, i.e. it becomes the
canonical production dataset that the rest of the workflow consumes; the order-2
run goes to a separate directory and exists only for the timing comparison.

Two components make up the cost:
  * the CFL tightens by kappa(6)/kappa(2) = 1.39907/1.25886 = 1.111x, so 11.1 %
    more time steps;
  * each update evaluates 6 stencil taps per derivative instead of 2.
The product is what is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.headless import (  # noqa: E402
    SetupParams, build_forward_inputs, run_forward,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 6])
    ap.add_argument("--canonical-order", type=int, default=6,
                    help="the order whose run lands in workspace/2D/forward")
    ap.add_argument("--out", default="workspace/2D/production_timing.json")
    args = ap.parse_args()

    results = {}
    for order in args.orders:
        run_dir = (Path("workspace/2D/forward") if order == args.canonical_order
                   else Path(f"workspace/2D/timing_order{order}"))
        meta = build_forward_inputs(run_dir, replace(SetupParams(), fd_order=order))
        timing = run_forward(run_dir, nproc=args.nproc)
        results[str(order)] = {
            "dir": str(run_dir), "nt": meta["nt_model"], "dt_s": meta["dt_model_target_s"],
            "dx_m": meta["dx_model_target_m"], "ntx": meta["ntx"],
            "wall_s": timing["wall_s"], "nproc": args.nproc,
        }
        print(f"[timing] order {order}: {timing['wall_s']:.1f} s "
              f"({timing['wall_s']/60:.2f} min) for {meta['ntx']} transmitters, "
              f"nt={meta['nt_model']:,}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")

    if "2" in results and "6" in results:
        a, b = results["2"], results["6"]
        step_ratio = b["nt"] / a["nt"]
        wall_ratio = b["wall_s"] / a["wall_s"]
        print(f"\n=== measured cost of order 2 -> order 6 ===")
        print(f"  time steps      : {a['nt']:,} -> {b['nt']:,}   ({step_ratio:.3f}x, "
              f"the CFL/kappa term)")
        print(f"  wall clock      : {a['wall_s']:.1f} s -> {b['wall_s']:.1f} s   "
              f"({wall_ratio:.2f}x)")
        print(f"  per time step   : {wall_ratio/step_ratio:.2f}x  (the wider stencil alone)")
        print(f"  This is a MEASURED number on {a['ntx']} transmitters at -np {args.nproc}, "
              f"not an extrapolation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
