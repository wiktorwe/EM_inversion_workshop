"""Task 2 A/B: does splitting the band into per-frequency runs reproduce the
broadband calibration, and what does the per-frequency grid do to interfaces?

Runs the lateral-average FDTD-vs-analytic calibration once per frequency, each
on that frequency's OWN designed grid (its own dx, dt, eps_r, PML), and compares
the assembled table against the single broadband run's.

Three things are checked, because each can fail independently:

1. **C reproduction.** The calibration already fits C per frequency, so it maps
   onto the split naturally - but it now runs once per frequency GRID rather
   than once for the band. If the assembled table does not reproduce the
   broadband one to within the existing scatter, that is either a real
   discretisation dependence (different dx really does give a different C) or a
   bug in the assembly, and this reports which: a discretisation dependence
   shows up as |C|/dx^2 tracking dx monotonically, a bug does not.

2. **Phase reference.** In the combined run all four tones share one time base,
   so relative phase between frequencies is automatic. With separate runs each
   extraction divides by that run's OWN source wavelet phasor, which should make
   the result independent of the time base. That is confirmed two ways: the
   per-frequency C phases are compared against the broadband ones (known good
   against the analytic solver), and the extraction is re-run on the same traces
   with the analysis window shifted, which must leave the gain unchanged because
   the shift multiplies trace and wavelet phasors by the same factor.

3. **Interface quantisation.** Each frequency now gets a different grid, so the
   resistivity model is resampled differently and interface positions quantise
   differently: an interface that lands on a cell face at 0.8 m generally will
   not at 1.6 m. The resulting per-frequency depth bias is measured here rather
   than left to hide inside C.
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
    METHOD_LATERAL_AVERAGE, lateral_average_resistivity_profile,
)
from scripts.modules.headless import (  # noqa: E402
    SetupParams, build_forward_inputs, build_per_frequency_forward_inputs, run_calibration,
)
from scripts.modules.rockem_bridge import steady_state_phasor  # noqa: E402


def interface_quantisation(fwd_dirs: dict, reference_dir: Path) -> dict:
    """Where the lateral-average model's interfaces land on each grid.

    The continuous interface depths come from the SEG-Y model's own 1 m
    sampling; each FD grid can only place an interface on one of its own cell
    boundaries, so the bias is up to half a cell and it is DIFFERENT per
    frequency once each frequency has its own dx.
    """
    z_ref, rho_ref = lateral_average_resistivity_profile(Path(reference_dir) / "sg.rss")
    out = {}
    for key, d in fwd_dirs.items():
        z, rho = lateral_average_resistivity_profile(Path(d) / "sg.rss")
        dz = float(np.median(np.diff(np.sort(z))))
        # interface = a cell boundary where resistivity changes
        idx = np.where(np.abs(np.diff(rho)) > 1e-9 * np.maximum(1.0, np.abs(rho[:-1])))[0]
        depths = 0.5 * (z[idx] + z[idx + 1]) if idx.size else np.array([])
        idx_r = np.where(np.abs(np.diff(rho_ref)) > 1e-9 * np.maximum(1.0, np.abs(rho_ref[:-1])))[0]
        depths_r = 0.5 * (z_ref[idx_r] + z_ref[idx_r + 1]) if idx_r.size else np.array([])
        shifts = []
        for d0 in depths_r:
            if depths.size:
                shifts.append(float(np.min(np.abs(depths - d0))))
        out[key] = {"dz_m": dz, "n_interfaces": int(depths.size),
                    "max_shift_m": float(max(shifts)) if shifts else None,
                    "rms_shift_m": float(np.sqrt(np.mean(np.square(shifts)))) if shifts else None,
                    "half_cell_m": 0.5 * dz}
    return out


def window_stability(run_dir: Path, freq_hz: float, n_periods_ref: float) -> dict:
    """Is the extracted gain stable against the choice of analysis window?

    `steady_state_phasor` references phase to the START OF ITS OWN WINDOW, and
    the same window is applied to the trace and to the injected wavelet, so a
    common time origin cancels in the ratio. That is the mechanism by which
    per-frequency runs share a phase reference WITHOUT sharing a time base.

    A test that only trims a prefix shorter than (record - window) changes
    nothing at all, because the extractor uses the LAST `n_periods` periods -
    it would pass no matter what. The test that can actually fail is varying
    the WINDOW LENGTH: if the gain moves, the window is picking up something
    that is not the steady-state response (in practice, the source's ramp-up).

    Reported relative to `n_periods_ref`, the value Step 01 wrote.
    """
    from scripts.modules.fd_visualization import load_rss_traces
    cal_dir = Path(run_dir) / "calibration_lateral_average"
    hx = load_rss_traces(cal_dir / "Data" / "Hxshot.rss")
    wav = load_rss_traces(cal_dir / "wav2d.rss")
    tr = np.asarray(hx["data"], dtype=float)[:, 0]
    w = np.asarray(wav["data"], dtype=float)[:, 0]

    def gain(npx):
        return (steady_state_phasor(tr, hx["dt"], freq_hz, npx)
                / steady_state_phasor(w, wav["dt"], freq_hz, npx))

    base = gain(n_periods_ref)
    out = {"n_periods_ref": float(n_periods_ref),
           "base_phase_deg": float(np.angle(base, deg=True)), "windows": []}
    for npx in (n_periods_ref - 2, n_periods_ref - 1, n_periods_ref,
                n_periods_ref + 1):
        if npx < 1:
            continue
        g = gain(npx)
        out["windows"].append({
            "n_periods": float(npx),
            "amp_rel_change": float(abs(abs(g) / abs(base) - 1.0)),
            "phase_change_deg": float(np.angle(g / base, deg=True)),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--root", default="workspace/2D/per_frequency")
    ap.add_argument("--broadband-dir", default="workspace/2D/forward")
    ap.add_argument("--out", default="workspace/2D/per_frequency/per_frequency_ab.json")
    args = ap.parse_args()

    root = Path(args.root)
    manifest = build_per_frequency_forward_inputs(root, SetupParams())

    per_freq = {}
    for key, run in manifest["runs"].items():
        cal = run_calibration(run["run_dir"], method=METHOD_LATERAL_AVERAGE,
                              nproc=args.nproc, save_to_metadata=True)
        c = np.asarray(cal["C_hxhz_shared"], dtype=complex)
        per_freq[key] = {
            "freq_hz": run["freq_hz"], "dx_m": float(cal["dx_m"]),
            "nt": run["meta"]["nt_model"],
            "C_over_dx2": float(np.abs(c[0]) / float(cal["dx_squared"])),
            "C_phase_deg": float(np.angle(c[0], deg=True)),
            "scatter_hx_pct": float(cal["scatter_hx_pct"][0]),
            "scatter_hz_pct": float(cal["scatter_hz_pct"][0]),
            "wall_s": cal["fdtd_wall_s"],
        }

    bb_path = Path(args.broadband_dir) / "setup_metadata.json"
    bb = json.loads(bb_path.read_text()).get("fdtd_analytic_calibration_by_source", {}).get("HX")
    if bb is None:
        bb = json.loads(bb_path.read_text()).get("fdtd_analytic_calibration")

    print("\n=== per-frequency runs vs the single broadband run "
          "(lateral-average Earth, same order) ===")
    print(f"{'f [Hz]':>8} {'dx m':>6} {'nt':>9} | {'|C|/dx2 split':>14} {'broadband':>10} "
          f"{'diff %':>8} | {'phase split':>12} {'broadband':>10} {'diff deg':>9} | "
          f"{'Hx scat %':>10}")
    rows = []
    for key in sorted(per_freq, key=lambda k: per_freq[k]["freq_hz"]):
        p = per_freq[key]
        i = int(np.argmin(np.abs(np.asarray(bb["freqs_hz"]) - p["freq_hz"])))
        bb_c = float(bb["C_over_dx_squared"][i]); bb_ph = float(bb["C_phase_deg"][i])
        d_amp = 100.0 * (p["C_over_dx2"] / bb_c - 1.0)
        d_ph = p["C_phase_deg"] - bb_ph
        print(f"{p['freq_hz']:8.0f} {p['dx_m']:6.2f} {p['nt']:9,} | {p['C_over_dx2']:14.5f} "
              f"{bb_c:10.5f} {d_amp:+8.3f} | {p['C_phase_deg']:+12.4f} {bb_ph:+10.4f} "
              f"{d_ph:+9.4f} | {p['scatter_hx_pct']:10.4f}")
        rows.append({"freq_hz": p["freq_hz"], "amp_diff_pct": d_amp, "phase_diff_deg": d_ph,
                     "scatter_hx_pct": p["scatter_hx_pct"], "bb_scatter_hx_pct":
                     float(bb["scatter_hx_pct"][i])})

    worst_amp = max(abs(r["amp_diff_pct"]) for r in rows)
    worst_ph = max(abs(r["phase_diff_deg"]) for r in rows)
    scat = max(max(r["scatter_hx_pct"], r["bb_scatter_hx_pct"]) for r in rows)
    print(f"\n  worst |C| difference : {worst_amp:.3f} %   (existing Hx scatter: {scat:.3f} %)")
    print(f"  worst phase difference: {worst_ph:.4f} deg")
    # THE CONTROL, and it is what actually discriminates a real discretisation
    # dependence from an assembly bug: the HIGHEST frequency's per-frequency run
    # uses the SAME grid as the broadband run (dx is set by f_max in both), so if
    # the assembly were wrong it would differ there too. A monotonic-in-dx test
    # is NOT the right discriminator - interface quantisation shifts by up to
    # half a cell in EITHER direction, so a real effect need not be monotonic.
    ctrl = min(rows, key=lambda r: abs(r["freq_hz"] - max(x["freq_hz"] for x in rows)))
    print(f"\n  CONTROL - highest frequency shares the broadband grid: "
          f"{abs(ctrl['amp_diff_pct']):.4f} % difference")
    if abs(ctrl["amp_diff_pct"]) <= max(ctrl["scatter_hx_pct"], ctrl["bb_scatter_hx_pct"]):
        print("  => the assembly is CORRECT (same grid reproduces the broadband value).")
        if worst_amp > scat:
            print(f"     The remaining differences up to {worst_amp:.3f} % are therefore a REAL")
            print("     dependence on the grid each frequency was designed for, not a bug.")
            print("     Compare them against the interface-quantisation table below: a coarser")
            print("     grid both resolves the near-source geometry with fewer cells per offset")
            print("     AND places the Earth's interfaces up to half a cell differently.")
    else:
        print("  => the assembly is SUSPECT: the same grid does not reproduce the "
              "broadband value.")
    dxs = [per_freq[k]["dx_m"] for k in sorted(per_freq, key=lambda k: per_freq[k]["freq_hz"])]
    amps = [r["amp_diff_pct"] for r in rows]
    print(f"     dx per run     : {[round(d, 3) for d in dxs]}")
    print(f"     |C| diff %     : {[round(a, 3) for a in amps]}")
    print(f"     offsets in cells: {[round(13.1 / d, 1) for d in dxs]}")

    print("\n=== phase reference: is the gain stable against the analysis window? ===")
    print("Each per-frequency run divides by its OWN wavelet phasor, and both are")
    print("windowed identically, so the time base cancels. What can still go wrong is")
    print("the window reaching into the source's ramp-up - which is what varying the")
    print("window LENGTH tests. Measured on the lowest-frequency run:")
    lo = min(per_freq, key=lambda k: per_freq[k]["freq_hz"])
    npx_ref = float(json.loads((Path(manifest["runs"][lo]["run_dir"])
                                / "setup_metadata.json").read_text())["n_periods_extract"])
    inv = window_stability(Path(manifest["runs"][lo]["run_dir"]),
                           per_freq[lo]["freq_hz"], npx_ref)
    for wrow in inv["windows"]:
        print(f"  n_periods = {wrow['n_periods']:4.1f}: "
              f"amplitude change {100*wrow['amp_rel_change']:8.4f} %, "
              f"phase change {wrow['phase_change_deg']:+8.4f} deg"
              + ("   <- the value Step 01 wrote" if wrow["n_periods"] == npx_ref else ""))

    print("\n=== interface quantisation per grid ===")
    print("Each frequency resamples the model onto its own dx, so interfaces land")
    print("differently. This is the depth bias that would otherwise hide inside C.")
    dirs = {k: v["run_dir"] for k, v in manifest["runs"].items()}
    dirs["broadband"] = str(args.broadband_dir)
    q = interface_quantisation(dirs, Path(args.broadband_dir))
    print(f"{'run':>12} {'dz m':>7} {'half cell':>10} {'max shift':>10} {'rms shift':>10}")
    for k, v in q.items():
        print(f"{k:>12} {v['dz_m']:7.2f} {v['half_cell_m']:10.3f} "
              f"{(v['max_shift_m'] if v['max_shift_m'] is not None else float('nan')):10.3f} "
              f"{(v['rms_shift_m'] if v['rms_shift_m'] is not None else float('nan')):10.3f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_freq": per_freq, "comparison": rows,
                               "window_stability": inv, "quantisation": q}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
