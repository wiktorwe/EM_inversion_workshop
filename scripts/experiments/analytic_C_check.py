"""Does the COMPUTED C(f) match the one an FDTD calibration fits?

This is the falsification test for `scripts/modules/fd_error_model.py`. `C` is
no longer fitted in the production path - it is `dx*dz*s(order)`, derived from
how the engine injects a source into one cell. That derivation is either right
or it is not, and this is what decides.

    python scripts/experiments/analytic_C_check.py

Reports, per dataset and per source, the fitted |C| against the computed one and
the deviation. The pass band is 0.5 %: an order below the smallest REAL term in
the error budget (the (dx/r)^2 geometric term is 0.37 % at the near offset), so
a deviation inside it cannot matter to an inversion, and one outside it means
the derivation is missing something.

A phase check comes free: the computed C is REAL, so a fitted phase that is not
small is a finding in its own right.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.fd_error_model import analytic_C, stencil_consistency  # noqa: E402
from scripts.modules.headless import iter_datasets  # noqa: E402

PASS_BAND_PCT = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-dir", default="workspace/2D/forward")
    ap.add_argument("--out", default="workspace/2D/analytic_C_check.json")
    ap.add_argument("--pass-band-pct", type=float, default=PASS_BAND_PCT)
    args = ap.parse_args()

    datasets = iter_datasets(Path(args.forward_dir))
    if not datasets:
        print(f"No datasets under {args.forward_dir}. Run Step 01 first.", file=sys.stderr)
        return 1

    rows, worst = [], 0.0
    print(f"{'dataset':>14} {'src':>4} {'f [Hz]':>8} {'dx':>6} "
          f"{'|C| fitted':>11} {'|C| computed':>13} {'dev %':>8} {'phase deg':>10}")
    for d in datasets:
        meta = json.loads((Path(d["run_dir"]) / "setup_metadata.json").read_text())
        dx = float(meta["dx_model_target_m"])
        order = int(meta["fd_order"])
        by_src = meta.get("fdtd_analytic_calibration_by_source") or {}
        if not by_src:
            print(f"{d['name']:>14}  (no calibration block - run Step 02)")
            continue
        for src, block in sorted(by_src.items()):
            freqs = np.asarray(block["freqs_hz"], dtype=float)
            C = (np.asarray(block["C_hxhz_shared_real"], dtype=float)
                 + 1j * np.asarray(block["C_hxhz_shared_imag"], dtype=float))
            want = analytic_C(dx, order)
            for i, f in enumerate(freqs):
                dev = 100.0 * (abs(C[i]) / want - 1.0)
                pha = float(np.degrees(np.angle(C[i])))
                worst = max(worst, abs(dev))
                rows.append({"dataset": d["name"], "source": src, "f_hz": float(f),
                             "dx_m": dx, "fd_order": order,
                             "C_fitted_abs": float(abs(C[i])), "C_analytic": float(want),
                             "deviation_pct": float(dev), "phase_deg": pha})
                print(f"{d['name']:>14} {src:>4} {f:8.0f} {dx:6.2f} "
                      f"{abs(C[i]):11.6f} {want:13.6f} {dev:+8.3f} {pha:+10.4f}")

    if not rows:
        print("\nNothing to check - no calibration blocks found.", file=sys.stderr)
        return 1

    ok = worst <= args.pass_band_pct
    worst_phase = max(abs(r["phase_deg"]) for r in rows)
    print(f"\ns(order) = {stencil_consistency(rows[0]['fd_order']):.6f}")
    print(f"worst |deviation| : {worst:.3f} %  (pass band {args.pass_band_pct:g} %)")
    print(f"worst |phase|     : {worst_phase:.4f} deg  (computed C is real)")
    print("RESULT:", "PASS - the computed C reproduces the fitted one" if ok else
          f"FAIL - {worst:.3f} % exceeds the pass band; the derivation is incomplete")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"rows": rows, "worst_deviation_pct": worst, "worst_phase_deg": worst_phase,
         "pass_band_pct": args.pass_band_pct, "pass": bool(ok)}, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
