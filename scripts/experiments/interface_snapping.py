"""A bias the calibration cannot see: continuous vs grid-quantised interfaces.

The 1D inversion fits an analytic forward whose layer interfaces can sit at ANY
depth, against FDTD data whose interfaces are quantised onto the FD grid (0.8 m
for the shipped design). That is a systematic depth error of up to half a cell,
and `C(f)` cannot absorb it: `C` is one complex number per frequency, shared by
every transmitter, whereas this error is model- and depth-dependent.

This script measures the bias in the units that matter - data change relative to
the calibration's own noise floor - and answers the practical question: is
half a cell of depth ambiguity above or below the level at which the inversion
can tell the difference?

Two numbers are reported:

* **actual quantisation** - where the true model's interfaces land after
  resampling from the SEG-Y 1 m sampling onto the FD grid, i.e. the depth error
  really present in the data being inverted;
* **data sensitivity** - the change in |Hx| and |Hz| produced by moving an
  interface by half a cell, compared against the calibration's Hx/Hz residual
  scatter and against the 3 % relative-error floor the inversion actually uses.

If the data change from half a cell is far below the floor, quantifying it is
enough and snapping is unnecessary. If it is comparable or larger, snapping the
inversion's candidate interfaces to cell faces is the fix -
`inversion_1d.unpack_model_params` takes a `snap_dz` argument for exactly that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.analytic_1d_forward import Layer1D, forward_1d_gains  # noqa: E402
from scripts.modules.fdtd_analytic_calibration import load_setup_metadata  # noqa: E402
from scripts.modules.multiscale_2d import read_sg_grid  # noqa: E402
from scripts.modules.segy import read_resistivity_from_segy  # noqa: E402


def interface_depths(z, rho):
    z = np.asarray(z, float); rho = np.asarray(rho, float)
    o = np.argsort(z); z, rho = z[o], rho[o]
    idx = np.where(np.abs(np.diff(rho)) > 1e-9 * np.maximum(1.0, np.abs(rho[:-1])))[0]
    return 0.5 * (z[idx] + z[idx + 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-dir", default="workspace/2D/forward")
    ap.add_argument("--segy", default="examples/Fault_1.sgy")
    ap.add_argument("--out", default="workspace/2D/interface_snapping.json")
    args = ap.parse_args()

    fwd = Path(args.forward_dir)
    meta = load_setup_metadata(fwd / "setup_metadata.json")
    dx = float(meta["dx_model_target_m"])
    eps_r = float(meta["eps_r_used"])
    freqs = np.asarray(meta["flist_hz"], float)
    tx_z = float(meta["tz0_m"])
    offs = np.asarray([float(meta["rx0_m"]) + i * float(meta["drx_m"])
                       for i in range(int(meta["nrx"]))])

    print(f"FD grid dx = dz = {dx:g} m  ->  half a cell = {dx/2:g} m")

    # --- 1. how badly are the TRUE interfaces quantised? --------------------
    seg = read_resistivity_from_segy(args.segy)
    z_seg = np.asarray(seg["z"], float)
    rho_seg = np.asarray(seg["resistivity"], float)          # (nz, nx)
    ix = int(np.argmin(np.abs(np.asarray(seg["x"], float) - float(meta["tx0_m"]))))
    d_true = interface_depths(z_seg, rho_seg[:, ix])
    # Compare the SAME x column, not the lateral average - averaging across the
    # fault throw mixes two different interface depths together and would be
    # reported as quantisation that is not there.
    g = read_sg_grid(fwd / "sg.rss")
    jx = int(np.argmin(np.abs(g["x"] - float(meta["tx0_m"]))))
    rho_col = 1.0 / np.clip(g["sigma"][jx, :], 1e-30, None)
    print(f"\n=== 1. interfaces as the FDTD actually sees them (column at tx0) ===")
    print(f"  SEG-Y model (dz = {seg['dz']:g} m): {np.round(d_true, 3)}")
    # sg.rss is built with LINEAR interpolation, so an interface is a one-cell
    # ramp rather than a step; its effective depth is the ramp's midpoint in
    # log10(rho), which is what a "where does the FDTD think the interface is"
    # question actually asks.
    lr = np.log10(rho_col)
    d_fd = []
    for d0 in d_true:
        w = np.where(np.abs(g["z"] - d0) <= 2.5)[0]
        if w.size < 3:
            continue
        # the interface is the steepest step inside the window; its effective
        # depth is where the ramp crosses the midpoint of the two plateaus
        k = int(w[np.argmax(np.abs(np.diff(lr[w])))])
        target = 0.5 * (lr[k] + lr[k + 1])
        d_fd.append(float(np.interp(target, sorted((lr[k], lr[k + 1])),
                                    (g["z"][k], g["z"][k + 1]) if lr[k] < lr[k + 1]
                                    else (g["z"][k + 1], g["z"][k]))))
    d_fd = np.asarray(d_fd)
    print(f"  FD grid     (dz = {dx:g} m), ramp midpoints: {np.round(d_fd, 3)}")
    shifts = [float(abs(d_fd[i] - d_true[i])) for i in range(min(d_fd.size, d_true.size))]
    if shifts:
        print(f"  interface depth shift: max {max(shifts):.3f} m, "
              f"rms {np.sqrt(np.mean(np.square(shifts))):.3f} m "
              f"(bound if it were a hard snap: half a cell = {dx/2:g} m)")
        print("  NOTE: sg.rss is resampled with LINEAR interpolation, so an interface is a")
        print("  ONE-CELL RAMP, not a step, and its effective depth is the ramp midpoint -")
        print("  which sits BETWEEN cell faces, not on one. Two consequences:")
        print("   * the mismatch is not bounded by half a cell in the tidy way a hard snap")
        print("     would be; it is whatever the ramp midpoint lands on;")
        print("   * snapping the inversion's candidate interfaces to cell FACES therefore")
        print("     only partly fixes it. To make snapping exact you would also have to")
        print("     resample the model blockily (nearest, not linear) so the FD interface")
        print("     really is on a face. Quantifying the bias, as below, is the honest")
        print("     option until that is done.")

    # --- 2. what does half a cell of depth do to the DATA? ------------------
    # Representative layered model: the 3 resistivities present in Fault_1,
    # with the transmitter inside the middle layer as in the real survey.
    base_thk = np.array([40.0, 35.0])
    rho = np.array([2.0, 25.0, 100.0])
    print(f"\n=== 2. data sensitivity to moving one interface ===")
    print(f"  model rho = {list(rho)} Ohm-m, thicknesses = {list(base_thk)} m, "
          f"tx at {tx_z:g} m, offsets {list(offs)} m")

    cal = meta.get("fdtd_analytic_calibration") or {}
    scat_hx = np.asarray(cal.get("scatter_hx_pct", [np.nan] * freqs.size), float)
    scat_hz = np.asarray(cal.get("scatter_hz_pct", [np.nan] * freqs.size), float)

    ref_hx, ref_hz = forward_1d_gains(rho, base_thk, freqs, offs, tx_z, tx_z, eps_r)
    rows = []
    print(f"\n{'shift m':>8} {'f [Hz]':>8} {'d|Hx| %':>9} {'d|Hz| %':>9} "
          f"{'Hx scatter %':>13} {'Hz scatter %':>13}")
    for shift in (dx / 2.0, dx, 2.0 * dx):
        thk = base_thk.copy(); thk[0] += shift
        hx, hz = forward_1d_gains(rho, thk, freqs, offs, tx_z, tx_z, eps_r)
        for i, f in enumerate(freqs):
            dhx = 100 * np.max(np.abs(np.abs(hx[i]) / np.abs(ref_hx[i]) - 1))
            dhz = 100 * np.max(np.abs(np.abs(hz[i]) / np.abs(ref_hz[i]) - 1))
            print(f"{shift:8.2f} {f:8.0f} {dhx:9.4f} {dhz:9.4f} "
                  f"{scat_hx[i]:13.4f} {scat_hz[i]:13.4f}")
            rows.append({"shift_m": float(shift), "f_hz": float(f),
                         "d_hx_pct": float(dhx), "d_hz_pct": float(dhz),
                         "scatter_hx_pct": float(scat_hx[i]),
                         "scatter_hz_pct": float(scat_hz[i])})

    half = [r for r in rows if abs(r["shift_m"] - dx / 2.0) < 1e-12]
    worst_hx = max(r["d_hx_pct"] for r in half)
    worst_hz = max(r["d_hz_pct"] for r in half)
    floor_pct = 3.0
    print(f"\n  half a cell ({dx/2:g} m) moves Hx by up to {worst_hx:.3f} % "
          f"and Hz by up to {worst_hz:.3f} %.")
    print(f"  The inversion's uncertainty floor is {floor_pct:g} % of |FDTD|, and the "
          f"calibration's own\n  residual scatter is "
          f"{np.nanmin(np.r_[scat_hx, scat_hz]):.3f}-{np.nanmax(np.r_[scat_hx, scat_hz]):.3f} %.")
    verdict = ("BELOW the noise floor - quantifying it is enough, snapping is optional"
               if max(worst_hx, worst_hz) < floor_pct else
               "AT OR ABOVE the noise floor - snap candidate interfaces to cell faces "
               "(inversion_1d.unpack_model_params(snap_dz=dx))")
    print(f"  => half-cell interface ambiguity is {verdict}.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"dx_m": dx, "true_interfaces_segy": d_true.tolist(),
         "fd_interfaces": d_fd.tolist(), "quantisation_shifts_m": shifts,
         "sensitivity": rows, "worst_half_cell_hx_pct": worst_hx,
         "worst_half_cell_hz_pct": worst_hz}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
