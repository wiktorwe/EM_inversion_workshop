"""Task 5: add a Kz magnetic line source alongside the existing Kx source.

The workshop hardcoded a single Kx source (`source_type=3`). Both receiver
components were already being recorded, so a second run with `source_type=5`
(Kz) completes the 2x2 magnetic coupling matrix:

    Kx -> Hx, Hz   (already had this)
    Kz -> Hx, Hz   (this is what is added)

With both source components in hand, ANY tilted transmitter and ANY tilted
receiver follow by exact superposition from the same two FDTD runs
(`tilted_magnetic_line_source_fields_layered`, `project_tilted_h`), so the
marginal cost per synthesised configuration is zero.

Three stages, run in this order, because each one is only interpretable if the
previous passed:

  --stage calibrate    Validate the new FDTD Kz run against
                       `magnetic_z_line_source_fields_layered` on the SAME
                       lateral-average and homogeneous Earths and with the SAME
                       complex least-squares C(f) fit as the Kx run. Expect
                       |C| ~ dx^2 and small phase, as for Kx. If the two source
                       types need DIFFERENT C, that is a source-normalisation
                       bug, not physics: both inject through the same dt/MU
                       coefficient into one cell.

  --stage reciprocity  Verify the Lorentz identity Hz(B | Kx@A) = Hx(A | Kz@B)
                       numerically on the LAYERED model, where the answer is
                       known, before touching the fault model - so indexing and
                       signs are checked in a case that cannot mislead. Uses a
                       purpose-built SYMMETRIC receiver line (the production
                       survey has only negative offsets, so it cannot form the
                       +-pair the identity needs).

  --stage fault        Run the real fault model for both sources, plot all four
                       couplings and the cross-term asymmetry as a function of
                       tool position, and report the DISTANCE at which each
                       observable departs from its background level by more
                       than the calibration scatter. That distance, not a
                       qualitative statement, is the answer to whether Kz makes
                       the method see further ahead.

Why the cross-term asymmetry is the interesting observable: in a purely layered
medium reciprocity FORCES Hz(B|Kx@A) = Hx(A|Kz@B), so their difference is
identically zero. It does not force them to agree when source and receiver see
different sides of a fault. The asymmetry is therefore an observable that is
exactly zero in the background and non-zero only in the presence of lateral
structure - a null background is the best place to look for a small signal.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.fd_visualization import compute_gains_for_fd_outputs  # noqa: E402
from scripts.modules.fdtd_analytic_calibration import (  # noqa: E402
    METHOD_HOMOGENEOUS, METHOD_LATERAL_AVERAGE,
)
from scripts.modules.headless import (  # noqa: E402
    SetupParams, build_forward_inputs, format_calibration_table, run_calibration, run_forward,
)


# ---------------------------------------------------------------------------
# stage: calibrate
# ---------------------------------------------------------------------------
def stage_calibrate(fwd_dir: Path, nproc: int, methods) -> dict:
    out = {}
    # Order matters: `save_calibration_to_metadata` puts the last HX calibration
    # into the ACTIVE slot that notebooks 05/06 read, so whichever Earth model
    # runs last wins. Run the LATERAL-AVERAGE one last, because that is the
    # calibration whose sigmas are on the same amplitude scale as the production
    # data the 1D inversion actually fits - the homogeneous rho_min sigmas are
    # nearly an order of magnitude smaller and would inflate every chi-squared.
    order = {METHOD_HOMOGENEOUS: 0, METHOD_LATERAL_AVERAGE: 1}
    methods = sorted(methods, key=lambda m: order.get(m, 99))
    for method in methods:
        for src in ("HX", "HZ"):
            cal = run_calibration(fwd_dir, method=method, nproc=nproc, source_field=src)
            c = np.asarray(cal["C_hxhz_shared"], dtype=complex)
            out[f"{method}_{src}"] = {
                "freqs_hz": list(cal["freqs_hz"]),
                "C_over_dx2": (np.abs(c) / float(cal["dx_squared"])).tolist(),
                "C_phase_deg": np.angle(c, deg=True).tolist(),
                "scatter_hx_pct": cal["scatter_hx_pct"],
                "scatter_hz_pct": cal["scatter_hz_pct"],
                "C_real": np.real(c).tolist(), "C_imag": np.imag(c).tolist(),
            }
    print("\n=== Kx vs Kz calibration constant ===")
    print("If the two source types need different C, that is a source-normalisation")
    print("bug, not physics - both inject through the same dt/MU into one cell.")
    for method in methods:
        a = out.get(f"{method}_HX"); b = out.get(f"{method}_HZ")
        if not (a and b):
            continue
        ca = np.asarray(a["C_real"]) + 1j * np.asarray(a["C_imag"])
        cb = np.asarray(b["C_real"]) + 1j * np.asarray(b["C_imag"])
        print(f"\n  {method}")
        print(f"    {'f [Hz]':>8} {'|C_Kx|/dx2':>11} {'|C_Kz|/dx2':>11} {'|Kz/Kx|':>9} {'arg(Kz/Kx)':>11}")
        for i, f in enumerate(a["freqs_hz"]):
            r = cb[i] / ca[i]
            print(f"    {f:8.0f} {a['C_over_dx2'][i]:11.5f} {b['C_over_dx2'][i]:11.5f} "
                  f"{abs(r):9.5f} {np.angle(r, deg=True):+10.4f}°")
    return out


# ---------------------------------------------------------------------------
# stage: reciprocity
# ---------------------------------------------------------------------------
# The production survey's receivers are all at NEGATIVE offsets (-13.1, -25.3 m),
# so it cannot form the +/- offset pair the reciprocity identity needs. This
# builds a symmetric receiver line on the same layered Earth instead.
RECIP_RX0_M = -25.3
RECIP_DRX_M = 6.325
RECIP_NRX = 9   # -25.3 ... +25.3 in 8 steps; the zero-offset one is dropped
                # by calibration_geometry_production (the field is singular there)


def stage_reciprocity(fwd_dir: Path, work_dir: Path, nproc: int) -> dict:
    """Hz(B | Kx@A) vs Hx(A | Kz@B) on the lateral-average layered Earth.

    In a laterally invariant (1D layered) medium, moving the source from A to B
    is a pure x-translation, so the identity reduces to a statement about the
    SAME run's offsets:

        Hz_from_Kx(+d)  ==  Hx_from_Kz(-d)

    at equal source and receiver depths. That is what is checked here.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((fwd_dir / "setup_metadata.json").read_text())
    meta["rx0_m"] = RECIP_RX0_M
    meta["drx_m"] = RECIP_DRX_M
    meta["nrx"] = RECIP_NRX
    (work_dir / "setup_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name in ("sg.rss", "wav2d.rss"):
        shutil.copy2(fwd_dir / name, work_dir / name)

    gains = {}
    for src in ("HX", "HZ"):
        cal = run_calibration(work_dir, method=METHOD_LATERAL_AVERAGE, nproc=nproc,
                              source_field=src, save_to_metadata=False)
        fdtd = cal["fdtd_result"]
        geo = fdtd["geometry"]
        off = np.asarray(geo["rx_x"], float) - float(np.asarray(geo["src_x"], float)[0])
        gains[src] = {
            "off": off,
            "hx": np.asarray(fdtd["Hx"]["gain"], complex),
            "hz": np.asarray(fdtd["Hz"]["gain"], complex),
            "freqs": np.asarray(cal["freqs_hz"], float),
            "analytic_hx": np.asarray(cal["analytic_hx"], complex),
            "analytic_hz": np.asarray(cal["analytic_hz"], complex),
            "C": np.asarray(cal["C_hxhz_shared"], complex),
        }

    freqs = gains["HX"]["freqs"]
    off_kx = gains["HX"]["off"]      # x-offsets of the Kx run
    off_kz = gains["HZ"]["off"]      # x-offsets of the Kz run (same survey)
    # Survey.rss stores coordinates as float32, so `+d` and `-d` do NOT come
    # back as exact negatives: at d = 25.3 m the round-trip leaves a residual of
    # ~7.6e-6 m. Matching on an exact 1e-6 tolerance therefore found NO pairs at
    # all. Match to a millimetre instead - far below the 0.8 m cell, so it
    # cannot pair the wrong receivers.
    match_tol_m = 1e-3
    print("\n=== Lorentz reciprocity on the layered Earth: Hz(Kx, +d) vs Hx(Kz, -d) ===")
    print("Analytic first (the identity must hold exactly there), then FDTD.")
    results = {"offsets": off_kx.tolist(), "freqs_hz": freqs.tolist(), "pairs": []}
    print("Residuals are reported TWO ways. Against the cross term itself they look")
    print("large, but the cross term IS the near-null - at 1 kHz it is only ~8 % of the")
    print("co-component. The meaningful normalisation is against |Hx(Kx)|, the strong")
    print("co-component, because that is the scale any fault signal is measured on.")
    print(f"{'f [Hz]':>8} {'+d [m]':>8} {'analytic':>10} {'FDTD/cross':>11} "
          f"{'FDTD/co-comp':>13} {'|cross|/|co|':>13}")
    for i, f in enumerate(freqs):
        for j, d in enumerate(off_kx):
            if d <= 0:
                continue
            k = int(np.argmin(np.abs(off_kz + d)))    # the -d receiver of the Kz run
            if abs(off_kz[k] + d) > match_tol_m:
                continue
            a = gains["HX"]["analytic_hz"][i, j]
            b = gains["HZ"]["analytic_hx"][i, k]
            an_err = abs(a - b) / max(abs(a), 1e-300)
            fa = gains["HX"]["hz"][i, j]
            fb = gains["HZ"]["hx"][i, k]
            fd_err = abs(fa - fb) / max(abs(fa), 1e-300)
            co = abs(gains["HX"]["hx"][i, j])          # the strong co-component
            co_err = abs(fa - fb) / max(co, 1e-300)
            print(f"{f:8.0f} {d:8.3f} {an_err:10.2e} {fd_err:11.3e} "
                  f"{co_err:13.3e} {abs(fa)/max(co,1e-300):13.4f}")
            results["pairs"].append({"f_hz": float(f), "offset_m": float(d),
                                     "analytic_rel_err": float(an_err),
                                     "fdtd_rel_err": float(fd_err),
                                     "fdtd_rel_err_vs_co": float(co_err),
                                     "cross_over_co": float(abs(fa) / max(co, 1e-300))})
    if not results["pairs"]:
        raise RuntimeError(
            f"No +d/-d receiver pairs matched within {match_tol_m} m. "
            f"Kx offsets: {np.round(off_kx, 4).tolist()}; "
            f"Kz offsets: {np.round(off_kz, 4).tolist()}. The reciprocity check needs a "
            "SYMMETRIC receiver line - check RECIP_RX0_M / RECIP_DRX_M / RECIP_NRX."
        )
    worst_an = max(p["analytic_rel_err"] for p in results["pairs"])
    worst_fd = max(p["fdtd_rel_err_vs_co"] for p in results["pairs"])
    print(f"\n  worst analytic reciprocity error : {worst_an:.2e}")
    print("    The identity itself is exact; this is the kx-quadrature floor of the")
    print("    primary/secondary decomposition, not a violation. (rockem-suite's own")
    print("    self-check reaches 4e-16 on a fixed-geometry analytic pair.)")
    print(f"  worst FDTD reciprocity error     : {worst_fd:.2e} of the co-component")
    print("    Discretisation-limited, and it tracks the cross component's own")
    print("    calibration scatter. THIS is the noise floor a fault-induced cross-term")
    print("    asymmetry has to beat - anything smaller is not a measurement.")
    results["worst_analytic"] = float(worst_an)
    results["worst_fdtd_vs_co"] = float(worst_fd)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", default=["calibrate"],
                    choices=["calibrate", "reciprocity", "fault"])
    ap.add_argument("--forward-dir", default="workspace/2D/forward")
    ap.add_argument("--work-dir", default="workspace/2D/hz_source")
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--methods", nargs="+",
                    default=[METHOD_LATERAL_AVERAGE, METHOD_HOMOGENEOUS])
    ap.add_argument("--out", default="workspace/2D/hz_source/hz_source.json")
    ap.add_argument("--fault-hx-dir", default=None,
                    help="existing Kx production run to reuse for --stage fault")
    ap.add_argument("--fault-hz-dir", default=None,
                    help="existing Kz production run to reuse for --stage fault")
    ap.add_argument("--fault-nproc", type=int, default=6)
    args = ap.parse_args()

    fwd = Path(args.forward_dir)
    work = Path(args.work_dir)
    out = {}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        out = json.loads(outp.read_text())

    if "calibrate" in args.stage:
        out["calibrate"] = stage_calibrate(fwd, args.nproc, args.methods)
        outp.write_text(json.dumps(out, indent=2) + "\n")
    if "reciprocity" in args.stage:
        out["reciprocity"] = stage_reciprocity(fwd, work / "reciprocity", args.nproc)
        outp.write_text(json.dumps(out, indent=2) + "\n")
    if "fault" in args.stage:
        from scripts.experiments.fault_couplings import stage_fault
        out["fault"] = stage_fault(work / "fault", args.fault_nproc,
                                   hx_dir=args.fault_hx_dir, hz_dir=args.fault_hz_dir)
        outp.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
