"""Cost of modelling the band in ONE broadband run vs one run per frequency.

Recomputes the design (it does not trust a table) by calling
``design_explicit_fd`` once per frequency with ``f_min = f_max = f``, and once
for the whole band, then compares total cell-steps.

Also reports what the two binding constraints cost:

* ``eps_r`` cap - ``eps_r`` is chosen as
  ``sigma_min / (tan_delta_floor * omega_max * eps0)``, so it GROWS as frequency
  falls and clips at ``eps_r_cap``. Since the explicit CFL ``dt`` scales like
  ``sqrt(eps_r)``, clipping throws away part of the low-frequency time-step
  gain. The cap is a round number, not a physical limit - the loss-tangent floor
  is what guarantees the displacement current stays negligible, and it is
  satisfied by construction at any of these values. Whether raising it is SAFE
  is a separate, measured question - see ``scripts/experiments/eps_r_bias.py``.
* minimum-offset rule - ``dx = min(induction limit, min_offset /
  cells_per_min_offset)``. At the low end the offset limit binds, so the
  low-frequency grid cannot coarsen as much as the physics alone would allow.
  That is correct and should NOT be relaxed: resolving the source-receiver
  separation is a geometric requirement independent of frequency.

Usage:
    python scripts/experiments/per_frequency_cost.py --orders 2 6
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.headless import SetupParams, fd_design_for  # noqa: E402


def cells(design: dict, depth_m: float, lpml: int) -> float:
    nx = design["apertx_m"] / design["dx_m"] + 2 * lpml
    nz = depth_m / design["dx_m"] + 2 * lpml
    return nx * nz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 6])
    ap.add_argument("--eps-r-caps", type=float, nargs="+", default=[1000.0, 1e9])
    ap.add_argument("--depth-m", type=float, default=120.0)
    args = ap.parse_args()

    base = SetupParams()
    n_periods = float(base.n_periods)
    flist = [float(f) for f in base.flist_hz]

    for order in args.orders:
        for cap in args.eps_r_caps:
            cap_label = "capped at 1000" if cap <= 1000.0 else "cap effectively removed"
            print(f"\n=== order {order}, eps_r {cap_label} ===")
            print(f"{'f kHz':>6} {'dx m':>6} {'eps_r':>8} {'cap?':>5} {'dt s':>11} "
                  f"{'nt':>8} {'cells':>8} {'cell-steps':>11}")
            total = 0.0
            for f in flist:
                p = replace(base, flist_hz=(f,), f_min_hz=f, f_max_hz=f,
                            fd_order=order, eps_r_cap=cap)
                d, _o, lpml = fd_design_for(p, max_depth_offset_m=0.0)
                nt = int((n_periods / f) / d["dt_s"]) + 1
                c = cells(d, args.depth_m, lpml)
                total += nt * c
                print(f"{f/1000:6.0f} {d['dx_m']:6.2f} {d['eps_r_used']:8.1f} "
                      f"{'yes' if d['eps_r_cap_binding'] else 'no':>5} {d['dt_s']:11.3e} "
                      f"{nt:8d} {c:8.0f} {nt*c:11.3e}")
            p = replace(base, fd_order=order, eps_r_cap=cap)
            d, _o, lpml = fd_design_for(p, max_depth_offset_m=0.0)
            nt = int((n_periods / min(flist)) / d["dt_s"]) + 1
            c = cells(d, args.depth_m, lpml)
            comb = nt * c
            print(f"{'sum':>6} {'':6} {'':8} {'':5} {'':11} {'':8} {'':8} {total:11.3e}")
            print(f"{'1 run':>6} {d['dx_m']:6.2f} {d['eps_r_used']:8.1f} "
                  f"{'yes' if d['eps_r_cap_binding'] else 'no':>5} {d['dt_s']:11.3e} "
                  f"{nt:8d} {c:8.0f} {comb:11.3e}")
            slowest = max(_per_run_cost(base, order, cap, f, args.depth_m) for f in flist)
            print(f"  serial     : {comb/total:.2f}x fewer cell-steps than the single broadband run")
            print(f"  concurrent : up to {comb/slowest:.2f}x less wall-clock "
                  f"({len(flist)} independent runs, limited by the slowest one)")
    return 0


def _per_run_cost(base: SetupParams, order: int, cap: float, f: float, depth_m: float) -> float:
    p = replace(base, flist_hz=(f,), f_min_hz=f, f_max_hz=f, fd_order=order, eps_r_cap=cap)
    d, _o, lpml = fd_design_for(p, max_depth_offset_m=0.0)
    nt = int((float(base.n_periods) / f) / d["dt_s"]) + 1
    return nt * cells(d, depth_m, lpml)


if __name__ == "__main__":
    raise SystemExit(main())
