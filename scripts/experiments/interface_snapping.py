"""How well do the FD model's interfaces line up with the true ones?

The 1D inversion fits an analytic forward whose layer interfaces can sit at ANY
depth, against FDTD data whose interfaces are quantised onto the FD grid. `C(f)`
cannot absorb that error: `C` is one complex number per frequency, shared by
every transmitter, whereas this error is model- and depth-dependent.

`sg.rss` is resampled from the SEG-Y model with NEAREST interpolation
(`headless.build_forward_inputs`, and notebook 01's `on_apply_outputs`), which
is what makes an interface a genuine step: its effective depth is the midpoint
of the two samples straddling it - i.e. exactly `oz + (k + 1/2) * dz`, so the
quantisation error is bounded by half a cell. LINEAR interpolation would make
every interface a ONE-CELL RAMP whose effective depth is the ramp midpoint, a
depth that sits between cell faces with no such bound, and snapping the
inversion's candidate interfaces to the grid could then only ever be a partial
fix.

This script measures, PER DATASET of the acquisition matrix (each has its own
grid, so each quantises differently):

* **actual quantisation** - where the true model's interfaces land in `sg.rss`,
  and whether they sit on the half-sample grid a nearest resample guarantees;
* **data sensitivity** - the change in |Hx| and |Hz| from moving an interface by
  half a cell, against the calibration's own residual scatter and the 3 %
  relative-error floor the inversion uses.

    python scripts/experiments/interface_snapping.py
    python scripts/experiments/interface_snapping.py --forward-dir workspace/2D/forward
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.analytic_1d_forward import forward_1d_gains  # noqa: E402
from scripts.modules.fdtd_analytic_calibration import load_setup_metadata  # noqa: E402
from scripts.modules.headless import iter_datasets  # noqa: E402
from scripts.modules.multiscale_2d import read_sg_grid  # noqa: E402
from scripts.modules.segy import read_resistivity_from_segy  # noqa: E402

FLOOR_PCT = 3.0          # fdtd_analytic_calibration.VALIDATED_REL_ERROR_FLOOR


def interface_depths(z, rho):
    z = np.asarray(z, float); rho = np.asarray(rho, float)
    o = np.argsort(z); z, rho = z[o], rho[o]
    idx = np.where(np.abs(np.diff(rho)) > 1e-9 * np.maximum(1.0, np.abs(rho[:-1])))[0]
    return 0.5 * (z[idx] + z[idx + 1])


def fd_interface_depths(sg_path: Path, tx0_m: float, d_true, window_m: float = 2.5):
    """Effective interface depths in `sg.rss`, read at the transmitter's column.

    The SAME x column as the true model, never the lateral average: averaging
    across the fault throw mixes two different interface depths and would be
    reported as quantisation that is not there.

    The effective depth is where the log-rho profile crosses the midpoint of the
    two plateaus either side of the steepest step. For a NEAREST resample that
    is exactly the midpoint of the two samples straddling the jump; the same
    formula reads the ramp midpoint for a legacy linear-resampled model, so the
    two are directly comparable.
    """
    g = read_sg_grid(sg_path)
    jx = int(np.argmin(np.abs(g["x"] - float(tx0_m))))
    lr = np.log10(1.0 / np.clip(g["sigma"][jx, :], 1e-30, None))
    z = np.asarray(g["z"], float)
    out, n_transition = [], []
    for d0 in d_true:
        w = np.where(np.abs(z - d0) <= window_m)[0]
        if w.size < 3:
            continue
        k = int(w[np.argmax(np.abs(np.diff(lr[w])))])
        target = 0.5 * (lr[k] + lr[k + 1])
        pair = (z[k], z[k + 1]) if lr[k] < lr[k + 1] else (z[k + 1], z[k])
        out.append(float(np.interp(target, sorted((lr[k], lr[k + 1])), pair)))
        # SHARPNESS. Count samples in the window strictly between the two
        # PLATEAU values - the window EDGES, not the two samples either side of
        # the steepest step. (Using the step's own endpoints measures nothing:
        # for a one-cell ramp the steepest step already lands on the ramp value,
        # so nothing is ever "between" them and every model looks sharp.)
        # A NEAREST resample gives 0 - the interface is a step. A LINEAR one
        # gives 1 or more: the FD medium at the interface is neither layer,
        # which is not what the analytic layered forward models.
        lo, hi = sorted((float(lr[w[0]]), float(lr[w[-1]])))
        tol = 1e-3 * max(hi - lo, 1e-12)
        n_transition.append(int(np.count_nonzero(
            (lr[w] > lo + tol) & (lr[w] < hi - tol))))
    return np.asarray(out), z, np.asarray(n_transition, dtype=int)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-dir", default="workspace/2D/forward",
                    help="Forward ROOT (matrix) or a single dataset directory.")
    ap.add_argument("--segy", default="examples/Fault_1.sgy")
    ap.add_argument("--out", default="workspace/2D/interface_snapping.json")
    args = ap.parse_args()

    datasets = iter_datasets(Path(args.forward_dir))
    if not datasets:
        print(f"No datasets under {args.forward_dir}. Run Step 01 first.", file=sys.stderr)
        return 1

    seg = read_resistivity_from_segy(args.segy)
    z_seg = np.asarray(seg["z"], float)
    rho_seg = np.asarray(seg["resistivity"], float)

    report = {"segy": args.segy, "datasets": []}
    for d in datasets:
        run = Path(d["run_dir"])
        meta = load_setup_metadata(run / "setup_metadata.json")
        dx = float(meta["dx_model_target_m"])
        eps_r = float(meta["eps_r_used"])
        freqs = np.asarray(meta["flist_hz"], float)
        tx0 = float(meta["tx0_m"])
        tx_z = float(meta["tz0_m"])
        offs = np.asarray([float(meta["rx0_m"]) + i * float(meta["drx_m"])
                           for i in range(int(meta["nrx"]))])

        ix = int(np.argmin(np.abs(np.asarray(seg["x"], float) - tx0)))
        d_true = interface_depths(z_seg, rho_seg[:, ix])
        d_fd, z_grid, n_trans = fd_interface_depths(run / "sg.rss", tx0, d_true)

        n = min(d_fd.size, d_true.size)
        shifts = np.abs(d_fd[:n] - d_true[:n])
        # A nearest resample puts every interface on the HALF-SAMPLE grid,
        # `oz + (k + 1/2)*dz`. Distance off that grid is the direct test of
        # whether the model is blocky or still ramped.
        oz, dzg = float(z_grid[0]), float(z_grid[1] - z_grid[0])
        half = ((d_fd - oz) / dzg - 0.5)
        off_grid = np.abs(half - np.round(half)) * dzg

        # --- data sensitivity, at THIS dataset's own tone and eps_r ---------
        base_thk = np.array([40.0, 35.0])
        rho = np.array([2.0, 25.0, 100.0])
        ref_hx, ref_hz = forward_1d_gains(rho, base_thk, freqs, offs, tx_z, tx_z, eps_r)
        thk = base_thk.copy(); thk[0] += dx / 2.0
        hx, hz = forward_1d_gains(rho, thk, freqs, offs, tx_z, tx_z, eps_r)
        d_hx = float(np.max([100 * np.max(np.abs(np.abs(hx[i]) / np.abs(ref_hx[i]) - 1))
                             for i in range(freqs.size)]))
        d_hz = float(np.max([100 * np.max(np.abs(np.abs(hz[i]) / np.abs(ref_hz[i]) - 1))
                             for i in range(freqs.size)]))

        cal = meta.get("fdtd_analytic_calibration") or {}
        row = {
            "dataset": d["name"], "source_field": d["source_field"],
            "freqs_hz": [float(f) for f in freqs], "dx_m": dx, "eps_r_used": eps_r,
            "true_interfaces_segy": d_true.tolist(),
            "fd_interfaces": d_fd.tolist(),
            "shift_rms_m": float(np.sqrt(np.mean(shifts ** 2))) if shifts.size else None,
            "shift_max_m": float(np.max(shifts)) if shifts.size else None,
            "half_cell_m": dx / 2.0,
            "off_half_sample_grid_max_m": float(np.max(off_grid)) if off_grid.size else None,
            "transition_cells_per_interface": n_trans.tolist(),
            "transition_cells_max": int(np.max(n_trans)) if n_trans.size else None,
            "half_cell_d_hx_pct": d_hx, "half_cell_d_hz_pct": d_hz,
            "scatter_hx_pct": cal.get("scatter_hx_pct"),
            "scatter_hz_pct": cal.get("scatter_hz_pct"),
        }
        report["datasets"].append(row)

        print(f"\n=== {d['name']}  (dx = {dx:g} m, half a cell = {dx/2:g} m, "
              f"eps_r = {eps_r:.1f}) ===")
        print(f"  true interfaces (SEG-Y, dz={seg['dz']:g} m): {np.round(d_true, 3)}")
        print(f"  FD interfaces  : {np.round(d_fd, 3)}")
        print(f"  placement error: rms {row['shift_rms_m']:.3f} m, "
              f"max {row['shift_max_m']:.3f} m   (hard-snap bound: {dx/2:g} m)")
        print(f"  distance off the half-sample grid: max "
              f"{row['off_half_sample_grid_max_m']:.4f} m "
              f"(this is the set of depths a snapped candidate can hit exactly)")
        print(f"  transition cells per interface : {n_trans.tolist()} "
              f"({'SHARP - a step, as the analytic forward assumes' if row['transition_cells_max'] == 0 else 'SMEARED - the FD medium at the interface is neither layer'})")
        print(f"  half a cell moves |Hx| by {d_hx:.3f} % and |Hz| by {d_hz:.3f} % "
              f"(uncertainty floor {FLOOR_PCT:g} %)")

    worst_shift = max((r["shift_max_m"] or 0.0) for r in report["datasets"])
    worst_off = max((r["off_half_sample_grid_max_m"] or 0.0) for r in report["datasets"])
    worst_trans = max((r["transition_cells_max"] or 0) for r in report["datasets"])
    worst_hx = max(r["half_cell_d_hx_pct"] for r in report["datasets"])
    worst_hz = max(r["half_cell_d_hz_pct"] for r in report["datasets"])
    report["summary"] = {
        "worst_shift_max_m": worst_shift,
        "worst_off_half_sample_grid_m": worst_off,
        "worst_transition_cells": worst_trans,
        "worst_half_cell_d_hx_pct": worst_hx,
        "worst_half_cell_d_hz_pct": worst_hz,
        "floor_pct": FLOOR_PCT,
    }
    print(f"\n=== summary over {len(report['datasets'])} dataset(s) ===")
    print(f"  worst interface placement error : {worst_shift:.3f} m")
    print(f"  worst distance off the half-sample grid: {worst_off:.4f} m")
    print(f"  worst transition cells per interface   : {worst_trans} "
          f"({'sharp steps' if worst_trans == 0 else 'smeared interfaces'})")
    print(f"  worst half-cell data change     : |Hx| {worst_hx:.3f} %, |Hz| {worst_hz:.3f} % "
          f"(floor {FLOOR_PCT:g} %)")
    print("  => " + ("half-cell ambiguity is BELOW the noise floor; quantifying it is enough"
                     if max(worst_hx, worst_hz) < FLOOR_PCT else
                     "half-cell ambiguity is AT OR ABOVE the noise floor; snapping candidate "
                     "interfaces onto each dataset's half-sample grid is the fix"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
