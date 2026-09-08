"""Run the production survey once per frequency and MEASURE the Task-2 saving.

`per_frequency_cost.py` predicts the saving from the design rules. This runs it:
the same 30-transmitter survey, once per frequency on that frequency's own grid,
timed against the single broadband run, and leaves the four datasets in place for
the multi-scale inversion ladder to consume.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.headless import (  # noqa: E402
    SetupParams, build_per_frequency_forward_inputs, load_manifest, run_forward,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="workspace/2D/per_frequency")
    ap.add_argument("--broadband-dir", default="workspace/2D/forward")
    ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--out", default="workspace/2D/per_frequency/production_timing.json")
    args = ap.parse_args()

    root = Path(args.root)
    manifest = (load_manifest(root) if (root / "manifest.json").exists()
                else build_per_frequency_forward_inputs(root, SetupParams()))

    results = {}
    total = 0.0
    for key, run in sorted(manifest["runs"].items(), key=lambda kv: kv[1]["freq_hz"]):
        d = Path(run["run_dir"])
        t = run_forward(d, nproc=args.nproc)
        results[key] = {"freq_hz": run["freq_hz"], "dir": str(d),
                        "dx_m": run["meta"]["dx_model_target_m"],
                        "nt": run["meta"]["nt_model"], "wall_s": t["wall_s"]}
        total += t["wall_s"]
        print(f"[perfreq] {run['freq_hz']:.0f} Hz: {t['wall_s']:.1f} s "
              f"(dx={run['meta']['dx_model_target_m']:.2f} m, "
              f"nt={run['meta']['nt_model']:,})", flush=True)

    bb = json.loads(Path("workspace/2D/production_timing.json").read_text()) \
        if Path("workspace/2D/production_timing.json").exists() else {}
    bb_wall = float(bb.get("6", {}).get("wall_s", 0.0))

    print("\n=== MEASURED cost of the per-frequency split (production survey) ===")
    print(f"{'f [Hz]':>8} {'dx m':>6} {'nt':>9} {'wall s':>9}")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["freq_hz"]):
        print(f"{v['freq_hz']:8.0f} {v['dx_m']:6.2f} {v['nt']:9,} {v['wall_s']:9.1f}")
    print(f"{'sum of 4':>8} {'':6} {'':9} {total:9.1f}")
    if bb_wall > 0:
        slowest = max(v["wall_s"] for v in results.values())
        print(f"{'1 run':>8} {'':6} {'':9} {bb_wall:9.1f}   (the single broadband run)")
        print(f"\n  serial     : {bb_wall/total:.2f}x less wall clock than the broadband run")
        print(f"  concurrent : up to {bb_wall/slowest:.2f}x, limited by the slowest "
              f"({max(results.values(), key=lambda v: v['wall_s'])['freq_hz']:.0f} Hz)")
        print("  Both are MEASURED on 30 transmitters, not extrapolated from the design rules.")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_frequency": results, "broadband_wall_s": bb_wall,
                               "sum_wall_s": total}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
