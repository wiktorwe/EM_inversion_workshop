"""The 2x2 magnetic coupling matrix across a fault, and how far ahead it sees.

Runs the production survey on `examples/Fault_1.sgy` TWICE - once with the Kx
line source (`source_type=3`) and once with Kz (`source_type=5`) - and forms all
four couplings per transmitter position:

    Cxx = Hx from Kx     Cxz = Hz from Kx
    Czx = Hx from Kz     Czz = Hz from Kz

plus the CROSS-TERM SUM

    S = Cxz + Czx

which is the interesting one. Derivation of why S is a null:

  1. Lorentz reciprocity gives  Hz(B | Kx@A) = Hx(A | Kz@B)  exactly
     (rockem-suite's own self-check verifies this to 4e-16).
  2. In a LATERALLY INVARIANT medium, swapping A and B is a pure x-translation
     plus an offset reversal, so  Hx(A | Kz@B) = Hx_from_Kz(offset = -d).
  3. At ZERO depth offset - this workshop's colinear survey - both cross
     components are ODD in offset, so Hx_from_Kz(-d) = -Hx_from_Kz(+d).

  => Hz_from_Kx(+d) = -Hx_from_Kz(+d), i.e. S = 0 identically, for the SAME
     source position and the SAME receiver, in ANY 1D layered Earth.

Verified numerically on a 3-layer stack: |S| / |Cxz| = 5e-16 at dz = 0, and
~2 (i.e. no null at all) as soon as dz != 0 - the depth symmetry is what makes
step 3 work, so this observable is specific to the colinear geometry.

S is therefore identically zero in the background model and non-zero ONLY where
lateral structure breaks the symmetry, and its sign says which side. A null
background is the best possible place to look for a small signal: the fault
appears against nothing rather than as a small perturbation on a large direct
coupling.

What is REPORTED, per observable and per frequency, is the tool position at
which it first departs from its own background level by more than the
calibration scatter - a distance in metres from the fault, not a qualitative
claim.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.modules.fd_visualization import compute_gains_for_fd_outputs
from scripts.modules.headless import SetupParams, build_forward_inputs, run_forward

# Vertical fault in examples/Fault_1.sgy: the layer stack steps from
# interfaces at z = 6020 / 6070 / 6080 m to 6030 / 6070 / 6090 m between the
# model columns at x = 1477 and 1479 m (segy dx = 2 m), i.e. a 10 m throw on the
# top and bottom interfaces with the middle one unfaulted.
FAULT_X_M = 1478.0

# ---------------------------------------------------------------------------
# APERTURE. Read this before believing any look-ahead distance from this script.
# ---------------------------------------------------------------------------
# `apertx > 0` is a source-centred TOTAL width, so each shot is modelled on a
# local model spanning only +-apertx/2 around its own transmitter. Structure
# further away than that is NOT IN THE MODELLED DOMAIN for that shot - it cannot
# be detected, no matter what the physics would do.
#
# The workshop's default `apertx = 2*max_offset + margin = 110.6 m` is sized from
# the SURVEY OFFSETS, which is right for the direct couplings and far too small
# for a look-ahead study: it gives a half-width of 55.3 m.
#
# Measured consequence, on the first run of this experiment: the observables were
# BIT-IDENTICAL at transmitters 78.0, 73.2, 68.4 and 63.6 m ahead of the fault
# (|S|/|Cxx| = 1.452e-04 at all four), because every one of those shots modelled
# the same laterally invariant Earth, and only began to move at 58.8 / 54.0 m -
# i.e. exactly at the 55.3 m aperture edge. The apparent "detection distance"
# was the aperture, not the physics. A physical response varies continuously
# with distance; a constant that switches on is a domain boundary.
#
# So: for a look-ahead experiment, set `apertx` from the LOOK-AHEAD RANGE you
# want to resolve (>= 2x the farthest transmitter-to-target distance), not from
# the survey offsets, and check that the farthest transmitters still show
# variation rather than a repeated constant. `stage_fault` warns when they do
# not.
MIN_LOOKAHEAD_APERTURE_MARGIN = 1.2


def per_tx_gains(run_dir: Path, meta: dict) -> dict:
    """Complex per-(freq, tx, local rx) channel gains from a production run."""
    g = compute_gains_for_fd_outputs(
        run_dir / "Data" / "Hxshot.rss",
        run_dir / "Data" / "Hzshot.rss",
        run_dir / "wav2d.rss",
        freqs=np.asarray(meta["flist_hz"], dtype=float),
        f_min_hz=float(meta["f_min_hz"]),
        n_periods_extract=float(meta["n_periods_extract"]),
    )
    geo = g["geometry"]
    tx_idx = np.asarray(geo["tx_idx_per_trace"], dtype=int)
    rx_loc = np.asarray(geo["rx_local_idx_per_trace"], dtype=int)
    tx_unique = np.asarray(geo["tx_unique"], dtype=float)
    nfreq = len(meta["flist_hz"])
    ntx = tx_unique.shape[0]
    nrx = int(rx_loc.max()) + 1

    hx = np.full((nfreq, ntx, nrx), np.nan, dtype=complex)
    hz = np.full((nfreq, ntx, nrx), np.nan, dtype=complex)
    off = np.full((ntx, nrx), np.nan, dtype=float)
    rx_x = np.asarray(geo["rx_x"], dtype=float)
    for tr in range(tx_idx.size):
        hx[:, tx_idx[tr], rx_loc[tr]] = g["Hx"]["gain"][:, tr]
        hz[:, tx_idx[tr], rx_loc[tr]] = g["Hz"]["gain"][:, tr]
        off[tx_idx[tr], rx_loc[tr]] = rx_x[tr] - tx_unique[tx_idx[tr], 0]
    return {"tx_x": tx_unique[:, 0], "off": off, "hx": hx, "hz": hz,
            "freqs": np.asarray(meta["flist_hz"], dtype=float)}


def _detection_distance(tx_x, values, threshold, fault_x=FAULT_X_M):
    """Distance from the fault at which |values| first exceeds `threshold`.

    `values` is a departure from background, already normalised. Transmitters
    are scanned from the FAR side inward, so the reported distance is the
    largest tool-to-fault distance at which the observable is still above
    threshold - i.e. how far ahead it can be seen, not where it peaks.
    """
    dist = fault_x - np.asarray(tx_x, dtype=float)   # >0 = tool before the fault
    v = np.abs(np.asarray(values, dtype=float))
    ok = np.isfinite(v) & (v > threshold)
    approaching = ok & (dist > 0)
    if not np.any(approaching):
        return None
    return float(np.max(dist[approaching]))


def ensure_production_run(run_dir: Path, source_field: str, *, nproc: int = 6,
                          verbose: bool = True, force: bool = False) -> dict:
    """Build + run one production forward, or reuse an existing complete one.

    Reuse is keyed on the recorded outputs existing AND the stored metadata
    agreeing on the source component, so a Kz directory can never be silently
    read as a Kx one.
    """
    run_dir = Path(run_dir)
    meta_path = run_dir / "setup_metadata.json"
    have = (run_dir / "Data" / "Hxshot.rss").exists() and (run_dir / "Data" / "Hzshot.rss").exists()
    if have and not force and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if str(meta.get("source_field", "HX")).upper() == source_field.upper():
            if verbose:
                print(f"[fault] reusing existing {source_field} run at {run_dir} "
                      f"(order={meta.get('fd_order')})")
            return {"dir": run_dir, "meta": meta, "wall_s": None, "reused": True}
    meta = build_forward_inputs(run_dir, replace(SetupParams(), source_field=source_field),
                                verbose=verbose)
    timing = run_forward(run_dir, nproc=nproc, verbose=verbose)
    return {"dir": run_dir, "meta": meta, "wall_s": timing["wall_s"], "reused": False}


def stage_fault(work_dir: Path, nproc: int = 6, verbose: bool = True,
                hx_dir: Path | None = None, hz_dir: Path | None = None) -> dict:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dirs = {"HX": Path(hx_dir) if hx_dir else work_dir / "hx",
            "HZ": Path(hz_dir) if hz_dir else work_dir / "hz"}
    runs = {src: ensure_production_run(dirs[src], src, nproc=nproc, verbose=verbose)
            for src in ("HX", "HZ")}

    gx = per_tx_gains(runs["HX"]["dir"], runs["HX"]["meta"])
    gz = per_tx_gains(runs["HZ"]["dir"], runs["HZ"]["meta"])
    assert np.allclose(gx["tx_x"], gz["tx_x"]), "Kx and Kz runs disagree on tx positions"

    couplings = {
        "Cxx": gx["hx"], "Cxz": gx["hz"],
        "Czx": gz["hx"], "Czz": gz["hz"],
    }
    couplings["S"] = couplings["Cxz"] + couplings["Czx"]

    # Guard against reporting the aperture as a detection distance (see the
    # APERTURE note at the top). If the observables do not vary across the
    # farthest transmitters, those shots do not contain the target at all.
    apertx = float(runs["HX"]["meta"]["apertx_m"])
    half_aperture = 0.5 * apertx
    tx_all = gx["tx_x"]
    dist_all = FAULT_X_M - tx_all
    outside = dist_all > half_aperture
    aperture_warning = None
    if np.any(outside):
        ref = np.abs(couplings["S"][-1, outside, 0])
        spread = float(np.ptp(ref) / max(np.mean(ref), 1e-300)) if ref.size > 1 else 0.0
        aperture_warning = (
            f"{int(outside.sum())} transmitter(s) are further from the fault than the "
            f"{half_aperture:.1f} m aperture half-width (apertx={apertx:.1f} m), so the "
            f"fault is OUTSIDE their local model. Relative spread of |S| across them: "
            f"{spread:.2e}"
            + (" - effectively constant, confirming they see no fault at all. Any "
               "'detection distance' at or beyond that radius is the APERTURE, not "
               "physics. Re-run with apertx >= 2x the look-ahead range you want."
               if spread < 1e-3 else ".")
        )
        print(f"\n*** APERTURE WARNING ***\n{aperture_warning}\n")
    out_aperture = {"apertx_m": apertx, "half_aperture_m": half_aperture,
                    "warning": aperture_warning}

    out = {
        "fault_x_m": FAULT_X_M,
        "aperture": out_aperture,
        "tx_x": gx["tx_x"].tolist(),
        "offsets_m": gx["off"][0].tolist(),
        "freqs_hz": gx["freqs"].tolist(),
        "wall_s": {k: v["wall_s"] for k, v in runs.items()},
        "observables": {},
    }
    for name, arr in couplings.items():
        out["observables"][name] = {
            "abs": np.abs(arr).tolist(),
            "phase_deg": np.angle(arr, deg=True).tolist(),
        }
    # --- background level and detection distance -------------------------
    # "Background" is taken from the transmitters FARTHEST from the fault on the
    # approach side, where the Earth beneath the tool is laterally invariant.
    # A departure counts once it exceeds the calibration's own residual scatter
    # for that component - the level below which FDTD and the analytic solver
    # already disagree, so nothing smaller is a measurement.
    tx_x = gx["tx_x"]
    dist = FAULT_X_M - tx_x
    far = dist >= np.percentile(dist[dist > 0], 60) if np.any(dist > 0) else np.zeros_like(dist, bool)
    cal = json.loads((Path(runs["HX"]["dir"]) / "setup_metadata.json").read_text()) \
        .get("fdtd_analytic_calibration_by_source", {}).get("HX") or {}
    scat_hx = np.asarray(cal.get("scatter_hx_pct", [np.nan] * len(out["freqs_hz"])), float) / 100.0
    scat_hz = np.asarray(cal.get("scatter_hz_pct", [np.nan] * len(out["freqs_hz"])), float) / 100.0
    thresholds = {"Cxx": scat_hx, "Cxz": scat_hz, "Czx": scat_hx, "Czz": scat_hz,
                  "S": np.minimum(scat_hx, scat_hz)}

    print("\n=== distance ahead of the fault at which each observable departs "
          "from background ===")
    print("Departure is measured against the CALIBRATION SCATTER for that component -")
    print("the level below which FDTD and the analytic solver already disagree.")
    print("S = Cxz + Czx is identically ZERO in a laterally invariant Earth, so it needs")
    print("no empirical background at all: it is normalised by |Cxx| (a real coupling)")
    print("and compared against the measured FDTD reciprocity floor.")
    print()
    print("CAVEAT, and it bounds what these numbers can mean: the survey spans only")
    print(f"{dist.max():.0f} m ahead of the fault to {-dist.min():.0f} m past it, while the skin depth at")
    print("28 Ohm-m runs 84.5 m (1 kHz) down to 34.5 m (6 kHz). At 1 and 2 kHz the")
    print("FARTHEST transmitter is still inside one skin depth of the fault, so the")
    print("'background' taken from those transmitters may already be contaminated and")
    print("the low-frequency detection distances below are LOWER BOUNDS - the survey")
    print("would have to be extended to measure them properly. At 4 and 6 kHz the far")
    print("transmitters are 2.0-2.3 skin depths out and the background is trustworthy.")
    detect = {}
    for name, arr in couplings.items():
        detect[name] = {}
        print(f"\n  {name}")
        print(f"    {'f [Hz]':>8} {'rx':>3} {'background':>12} {'threshold':>10} "
              f"{'detect at':>10}")
        for i, f in enumerate(out["freqs_hz"]):
            thr = float(thresholds[name][i]) if np.isfinite(thresholds[name][i]) else 0.03
            for r in range(arr.shape[2]):
                v = arr[i, :, r]
                if name == "S":
                    ref = np.abs(couplings["Cxx"][i, :, r])
                    normed = np.abs(v) / np.maximum(ref, 1e-300)
                    bg = float(np.nanmedian(normed[far])) if np.any(far) else np.nan
                    dep = normed - bg
                else:
                    bg = float(np.nanmedian(np.abs(v[far]))) if np.any(far) else np.nan
                    dep = np.abs(v) / max(bg, 1e-300) - 1.0
                d = _detection_distance(tx_x, dep, thr)
                detect[name][f"f{f:.0f}_rx{r}"] = d
                print(f"    {f:8.0f} {r:3d} {bg:12.5g} {thr:10.4f} "
                      f"{(f'{d:8.1f} m' if d is not None else '       -'):>10}")
    out["detection"] = detect
    out["background_tx_mask"] = far.tolist()
    (work_dir / "fault_couplings.json").write_text(json.dumps(out, indent=2) + "\n")
    return out
