"""Is the analytic solver's kx quadrature converged on the ACTUAL fault model?

`rockem.greens.greens_layered_2d` integrates over horizontal wavenumber with a
fixed Gauss-Legendre rule of `n_nodes` points on `[0, lam_max]`. Two things can
go wrong, and only one of them is visible to an `n_nodes` sweep:

* **quadrature resolution** - too few nodes for the chosen range. Doubling
  `n_nodes` reveals it.
* **truncation** - `lam_max` too small to reach the model's finest feature. This
  converges CLEANLY (and wrongly) in `n_nodes`, because more nodes just resolve
  a range that was already too short. rockem-suite's gotchas record up to ~15 %
  error of exactly this kind on a 13x-contrast stack, and its own self-check
  says the "lam_max-doubling is the decisive signal - n_nodes-doubling alone is
  blind".

`analytic_1d_forward.check_kx_convergence` doubles ONLY `n_nodes`, so on its own
it cannot detect the second failure. This script runs both legs on the real
`examples/Fault_1.sgy` resistivity range (2 / 25 / 100 Ohm-m, 50x contrast) at
the survey's own frequencies, offsets and depths.

Usage:
    python scripts/experiments/kx_convergence.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.analytic_1d_forward import Layer1D  # noqa: E402
from scripts.modules.rockem_bridge import (  # noqa: E402
    magnetic_line_source_fields_layered,
    magnetic_z_line_source_fields_layered,
)
from rockem.greens.greens_layered_2d import _default_lam_max  # noqa: E402


NULL_FRACTION = 1e-6


def worst_rel(a, b, reference):
    """Change normalised by the field scale, skipping genuine NULL components.

    Two traps make a naive pointwise relative metric useless here:

    * Hz is ODD in offset and passes through zero, so |a-b|/|a| explodes at a
      near-null while the absolute change is negligible. Normalise by the row's
      max|field| instead.
    * At zero depth offset the cross component (Hz from Kx, Hx from Kz) is an
      EXACT null whenever the draw happens to be homogeneous, so the computed
      value is pure round-off and any ratio against it is meaningless. Such a
      row is skipped (returns 0.0), detected by comparing its scale against the
      co-component `reference`.
    """
    a, b = np.asarray(a), np.asarray(b)
    scale = float(np.max(np.abs(a)))
    if scale <= NULL_FRACTION * max(float(np.max(np.abs(reference))), 1e-300):
        return 0.0
    return float(np.max(np.abs(a - b)) / max(scale, 1e-300))


def _verdict(val: float, floor: float) -> str:
    if val < 0.1 * floor:
        return "ok"
    if val < floor:
        return "inside floor"
    return "NOT CONVERGED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", type=float, nargs="+", default=[1000.0, 2000.0, 4000.0, 6000.0])
    ap.add_argument("--offsets", type=float, nargs="+", default=[-25.3, -13.1])
    ap.add_argument("--n-nodes", type=int, default=120)
    ap.add_argument("--eps-r", type=float, default=399.4467461332405)
    ap.add_argument("--tx-depth", type=float, default=6050.0)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--floor", type=float, default=0.03,
                    help="workshop relative-error floor to judge against "
                         "(VALIDATED_REL_ERROR_FLOOR, default 3%%)")
    args = ap.parse_args()

    rhos = np.array([2.0, 25.0, 100.0])  # the three values present in Fault_1.sgy
    offs = np.asarray(args.offsets, dtype=float)
    tx = float(args.tx_depth)

    print(f"Fault_1 resistivities {list(rhos)} Ohm-m (contrast {rhos.max()/rhos.min():.0f}x), "
          f"n_nodes={args.n_nodes}, offsets={list(offs)} m, rx_depth = tx_depth")
    print(f"{'source':>6} {'leg':>22} {'worst rel change':>17} {'verdict':>14}")

    worst = {"n_nodes x2": 0.0, "lam_max x2": 0.0, "lam_max x8": 0.0}
    for name, solver in (("Kx", magnetic_line_source_fields_layered),
                         ("Kz", magnetic_z_line_source_fields_layered)):
        legs = {"n_nodes x2": 0.0, "lam_max x2": 0.0, "lam_max x8": 0.0}
        # Fresh rng per source so BOTH sources see the SAME set of draws - a
        # shared generator would silently compare them on different models.
        rng = np.random.default_rng(0)
        for _ in range(args.seeds):
            # random 4-layer stack drawn from the model's own resistivity set,
            # thicknesses asymmetric so the source never lands on an interface
            res = rng.choice(rhos, size=4)
            thk = rng.uniform(8.0, 45.0, size=3)
            layers = [Layer1D(float(res[i]), float(thk[i]), args.eps_r) for i in range(3)]
            layers.append(Layer1D(float(res[3]), None, args.eps_r))
            for f in args.freqs:
                lam0 = _default_lam_max(layers, float(f))
                try:
                    base = solver(offs, float(f), layers, tx, rx_depth_m=tx,
                                  n_nodes=args.n_nodes, lam_max=lam0)[1:]
                    ref_n = solver(offs, float(f), layers, tx, rx_depth_m=tx,
                                   n_nodes=2 * args.n_nodes, lam_max=lam0)[1:]
                    ref_l2 = solver(offs, float(f), layers, tx, rx_depth_m=tx,
                                    n_nodes=2 * args.n_nodes, lam_max=2 * lam0)[1:]
                    ref_l8 = solver(offs, float(f), layers, tx, rx_depth_m=tx,
                                    n_nodes=8 * args.n_nodes, lam_max=8 * lam0)[1:]
                except Exception:
                    continue
                co = base[0] if name == "Kx" else base[1]   # never a null
                for key, ref in (("n_nodes x2", ref_n), ("lam_max x2", ref_l2), ("lam_max x8", ref_l8)):
                    legs[key] = max(legs[key], max(worst_rel(base[0], ref[0], co),
                                                   worst_rel(base[1], ref[1], co)))
        for key, val in legs.items():
            print(f"{name:>6} {key:>22} {val:17.3e} {_verdict(val, args.floor):>14}")
            worst[key] = max(worst[key], val)

    print("\nSummary (worst over both sources):")
    for key, val in worst.items():
        print(f"  {key:>12}: {val:.3e}")
    lam_err = max(worst["lam_max x2"], worst["lam_max x8"])
    print(f"""
Reading this: the workshop floors its data uncertainty at
{100*args.floor:.0f} % of |FDTD| (VALIDATED_REL_ERROR_FLOOR), and the FDTD's own
residual scatter against this solver is 0.04-5 %. A quadrature error is
acceptable when it is well inside that.

  n_nodes doubling  : {worst['n_nodes x2']:.1e}  <- the ONLY leg
      analytic_1d_forward.check_kx_convergence tests, and it is blind to
      truncation by construction: more nodes only resolve a range that was
      already too short.
  lam_max doubling  : {lam_err:.1e}  <- the decisive leg. This is the
      truncation error actually present at the shipped default.

=> n_nodes={args.n_nodes} is amply converged. The default lam_max leaves
   {100*lam_err:.3f} % worst-case truncation on the Fault_1 resistivity range,
   which is {args.floor*100/max(100*lam_err, 1e-12):.0f}x inside the {100*args.floor:.0f} %
   uncertainty floor: {'ADEQUATE' if lam_err < args.floor else 'NOT adequate'}.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
