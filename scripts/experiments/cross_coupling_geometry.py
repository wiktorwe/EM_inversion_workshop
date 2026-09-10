#!/usr/bin/env python3
"""Can a 1D layered model produce the cross-couplings Cxz / Czx?

It can, and the answer is dominated by ACQUISITION GEOMETRY rather than by the
model class. `Cxz` (Hz from a Kx line source) vanishes identically in one case
only: a HOMOGENEOUS whole space with the receiver at the source depth, which is
the on-axis direction of a magnetic line dipole. Two things break that null,
and this script measures both against the co-component `Cxx`:

* **layering** at zero depth offset - a few percent;
* **depth offset**, which lifts the ratio to order 1 even with no layering at
  all.

Why it matters: the workshop's shipped survey is colinear with `off_z = 0`
(receivers in the same horizontal well as the transmitter), so it sits exactly
on the null and reads the cross-couplings at a level where they carry almost no
1D information. That is a property of where the receivers are, not evidence
that a 1D model has nothing to say about these components.

    python scripts/experiments/cross_coupling_geometry.py
    python scripts/experiments/cross_coupling_geometry.py --freq 2000

The numbers this prints are the ones quoted in `KNOWN_ISSUES.md` section 5.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.modules.analytic_1d_forward import forward_1d_gains  # noqa: E402

TX_Z_M = 6050.0
EPS_R = 599.2
OFF_X_M = (-13.1, -25.3)
OFF_Z_M = (0.0, 0.5, 2.0, 5.0, 10.0, 20.0)

# (label, rho per layer, thickness of every layer but the halfspace)
EARTHS = (
    ("homogeneous 30", (30.0,), ()),
    ("layered 30/5/30", (30.0, 5.0, 30.0), (40.0, 15.0)),
    ("layered 30/5/100", (30.0, 5.0, 100.0), (40.0, 15.0)),
)


def ratios(rho, thk, freq_hz, off_z_m):
    """|Cxz/Cxx| and |Czx/Czz| at the near receiver, for one depth offset."""
    off_x = np.asarray(OFF_X_M, dtype=float)
    rx_z = TX_Z_M + float(off_z_m)
    kw = dict(freqs_hz=[float(freq_hz)], off_x=off_x, tx_depth_m=TX_Z_M,
              rx_depth_m=rx_z, eps_r=EPS_R)
    cxx, cxz = forward_1d_gains(np.asarray(rho, float), np.asarray(thk, float),
                                source_field="HX", **kw)
    czx, czz = forward_1d_gains(np.asarray(rho, float), np.asarray(thk, float),
                                source_field="HZ", **kw)
    return (abs(cxz[0, 0]) / abs(cxx[0, 0]),
            abs(czx[0, 0]) / abs(czz[0, 0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, default=4000.0, help="tone in Hz")
    ap.add_argument("--json", type=Path, default=None, help="also write the table here")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    print(f"\ncross-coupling size relative to its co-component, {args.freq:.0f} Hz, "
          f"near receiver at {OFF_X_M[0]} m\n")
    header = f"{'off_z (m)':>10} | " + " | ".join(f"{lab:>28}" for lab, _r, _t in EARTHS)
    print(header)
    print("-" * len(header))

    rows = []
    for off_z in OFF_Z_M:
        cells, rec = [], {"off_z_m": off_z}
        for lab, rho, thk in EARTHS:
            r_xz, r_zx = ratios(rho, thk, args.freq, off_z)
            cells.append(f"{r_xz:12.4e} {r_zx:12.4e}   ")
            rec[lab] = {"Cxz_over_Cxx": r_xz, "Czx_over_Czz": r_zx}
        print(f"{off_z:10.1f} | " + " | ".join(c.strip().rjust(28) for c in cells))
        rows.append(rec)
    print(f"\n{'':10}   each cell is |Cxz/Cxx|  then  |Czx/Czz|")

    homo_zero = rows[0][EARTHS[0][0]]["Cxz_over_Cxx"]
    lay_zero = rows[0][EARTHS[1][0]]["Cxz_over_Cxx"]
    deep = next(r for r in rows if r["off_z_m"] == 5.0)[EARTHS[0][0]]["Cxz_over_Cxx"]

    print("\nreading:")
    print(f"  homogeneous, off_z=0   : {homo_zero:.3e}  - the on-axis null, exactly zero")
    print(f"  layered,     off_z=0   : {lay_zero:.3e}  - layering alone breaks it")
    print(f"  homogeneous, off_z=5 m : {deep:.3e}  - geometry alone makes it order 1")

    ok = homo_zero < 1e-12 and lay_zero > 1e-3 and deep > 0.1
    print("\nRESULT: " + ("PASS - the cross-couplings are a 1D observable "
                          "wherever the receivers leave the source depth"
                          if ok else "FAIL - the expected ordering did not hold"))
    if args.json:
        args.json.write_text(json.dumps(
            {"freq_hz": args.freq, "tx_z_m": TX_Z_M, "off_x_m": list(OFF_X_M),
             "rows": rows}, indent=2))
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
