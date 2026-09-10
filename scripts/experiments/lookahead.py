"""How far ahead of the fault can each magnetic coupling actually see?

This measures what `fault_couplings.stage_fault` cannot: staged at survey
scale, that experiment measures the MODELLING APERTURE, not the physics (see the
APERTURE note in `fault_couplings.py`). Three things differ here:

1. **apertx is sized from the look-ahead range, not the survey offsets.**
   `apertx > 0` is a source-centred TOTAL width, so a fault further than
   `apertx/2` from a transmitter is not in that shot's local model at all. The
   default 110.6 m gives a 55.3 m half-width, and the observables are
   bit-identical at every transmitter beyond it.

2. **The transmitter line is extended** well past the intended detection range,
   so there are transmitters that genuinely see nothing and can serve as
   background.

3. **Each frequency runs on its own grid.** That is what makes this affordable:
   the record length follows each tone instead of the lowest one in the band,
   so `nt` stays near 33k per run rather than the ~197k a 1-6 kHz broadband run
   needs at the 6 kHz cell size.

WHAT THIS COSTS, measured on a 6-core run of the shipped configuration
(`apertx=400`, `ntx=32`): the 1 kHz stage takes **133 s**. Cost scales as
`ntx * (apertx/dx) * (1/dx) * nt`, which reproduces that measurement to better
than 1 % and the production matrix's own timings to ~15 %, giving

    1 / 2 / 4 / 6 kHz   ~2.2 / 2.3 / 5.3 / 7.3 min per (source, model)
    x 2 sources x 2 models (fault, reference)   ~= 69 min total

`fault_couplings.stage_fault` is two BROADBAND production runs, ~11 min each,
so the pair of experiments is about 90 minutes on this machine.

THE BACKGROUND IS A SECOND MODEL, NOT A FAR TRANSMITTER
-------------------------------------------------------
Taking the background from the transmitters farthest from the fault is
circular: "far enough that the signal has decayed" is exactly the quantity being
measured, and if `apertx` is what makes them quiet then the answer is the
aperture again. Worse, a background drawn from transmitters OUTSIDE the aperture
is null by construction.

So every configuration is run TWICE - once on `examples/Fault_1.sgy` and once on
`examples/Fault_1_nofault.sgy`, a laterally invariant model built by replicating
the fault model's own left-side column across the whole section. The two share
the survey, the grid, the wavelet and the FD design, so the ONLY difference is
the fault. The departure is then

    departure(tx) = | C_fault(tx) - C_nofault(tx) | / | C_nofault(tx) |

which is identically zero everywhere the fault has no influence, with no
assumption about where that is. The detection threshold is the calibration
scatter for that component - the level below which FDTD and the analytic solver
already disagree, so nothing smaller is a measurement.

The observables are the 2x2 magnetic coupling matrix plus the cross-term sum
`S = Cxz + Czx`, which is identically zero in any laterally invariant Earth at
zero depth offset (see `fault_couplings`'s derivation) - so on the reference
model `S` measures the FDTD's own reciprocity floor directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.experiments.fault_couplings import (  # noqa: E402
    FAULT_X_M, departure_thresholds, per_tx_gains)
from scripts.modules.headless import (  # noqa: E402
    SetupParams, build_forward_inputs, run_forward,
)


def plot_profiles(out: dict, path: Path) -> Path:
    """Each coupling and the cross-term sum vs distance ahead of the fault.

    One panel per frequency, log-scaled departure, with the detection threshold
    and the aperture half-width drawn in - the second because a curve that goes
    flat at the aperture is reporting the domain boundary, not the Earth, and
    that should be visible rather than inferred.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = sorted(out["observables"], key=float)
    if not freqs:
        raise ValueError("no observables to plot")
    fig, axes = plt.subplots(1, len(freqs), figsize=(4.2 * len(freqs), 4.0),
                             sharey=True, squeeze=False)
    colours = {"Cxx": "#1f77b4", "Cxz": "#d62728", "Czx": "#2ca02c",
               "Czz": "#9467bd", "S": "#000000"}
    for ax, f in zip(axes[0], freqs):
        obs = out["observables"][f]
        for name in ("Cxx", "Cxz", "Czx", "Czz", "S"):
            if name not in obs:
                continue
            d = np.asarray(obs[name]["dist_m"], float)
            v = np.asarray(obs[name]["departure"], float)
            m = d > 0
            ax.semilogy(d[m], np.maximum(v[m], 1e-12), label=name,
                        color=colours[name], lw=2.2 if name == "S" else 1.4,
                        zorder=3 if name == "S" else 2)
            ax.axhline(obs[name]["threshold"], color=colours[name], ls=":", lw=0.8, alpha=0.5)
        ax.axvline(out["half_aperture_m"], color="0.4", ls="--", lw=1.0)
        ax.text(out["half_aperture_m"], ax.get_ylim()[1], " aperture/2",
                rotation=90, va="top", ha="right", fontsize=7, color="0.4")
        ax.invert_xaxis()          # tool approaches the fault left-to-right
        ax.set_xlabel("distance ahead of the fault [m]")
        ax.set_title(f"{float(f):.0f} Hz")
        ax.grid(alpha=0.25, which="both")
    axes[0][0].set_ylabel("departure from the no-fault reference\n(fraction, dotted = threshold)")
    axes[0][-1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Magnetic couplings vs tool position - fault model minus "
                 "laterally invariant reference", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="workspace/2D/lookahead")
    ap.add_argument("--apertx", type=float, default=400.0,
                    help="source-centred TOTAL width; half of this is the max detectable range")
    ap.add_argument("--tx0", type=float, default=1318.0)
    ap.add_argument("--dtx", type=float, default=6.0)
    ap.add_argument("--ntx", type=int, default=32)
    ap.add_argument("--freqs", type=float, nargs="+", default=[1000.0, 2000.0, 4000.0, 6000.0])
    ap.add_argument("--segy-fault", default="examples/Fault_1.sgy")
    ap.add_argument("--segy-reference", default="examples/Fault_1_nofault.sgy")
    ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--out", default="workspace/2D/lookahead/lookahead.json")
    ap.add_argument("--figure", default="workspace/2D/lookahead/lookahead.png")
    ap.add_argument("--plot-only", action="store_true",
                    help="re-plot from a saved --out JSON without re-running any FDTD")
    args = ap.parse_args()

    if args.plot_only:
        out = json.loads(Path(args.out).read_text())
        print("wrote", plot_profiles(out, Path(args.figure)))
        return 0

    root = Path(args.root)
    half = 0.5 * args.apertx
    far = FAULT_X_M - args.tx0
    print(f"aperture half-width {half:.1f} m; farthest transmitter is {far:.1f} m "
          f"ahead of the fault at x={FAULT_X_M:.0f} m")
    if half < far:
        print(f"  WARNING: half-width {half:.1f} m < {far:.1f} m, so the farthest "
              f"transmitters still cannot see the fault. Raise --apertx.")

    results = {}
    for f in args.freqs:
        for src in ("HX", "HZ"):
            for tag, segy in (("fault", args.segy_fault), ("ref", args.segy_reference)):
                d = root / f"f{f:.0f}Hz_{src.lower()}_{tag}"
                p = replace(SetupParams(), segy_path=Path(segy),
                            flist_hz=(f,), f_min_hz=f, f_max_hz=f,
                            source_field=src, apertx_override_m=args.apertx,
                            tx0_m=args.tx0, dtx_m=args.dtx, ntx=args.ntx)
                meta = build_forward_inputs(d, p)
                t = run_forward(d, nproc=args.nproc)
                results[f"{f:.0f}_{src}_{tag}"] = {
                    "dir": str(d), "wall_s": t["wall_s"], "meta": meta}
                print(f"[lookahead] {f:.0f} Hz {src} {tag}: {t['wall_s']:.1f} s", flush=True)

    out: dict = {"fault_x_m": FAULT_X_M, "apertx_m": args.apertx,
                 "half_aperture_m": half, "observables": {}, "detection": {}}

    # Detection threshold: the modelling floor that survives a fault-vs-reference
    # difference, PER FREQUENCY, because each tone here is modelled on its own
    # grid and the floor is a grid property. See `departure_thresholds` - the
    # terms both runs share cancel out of the ratio, so this is the interface
    # quantisation and not the whole error budget.
    thresholds_by_freq = {}
    for f in args.freqs:
        meta = results[f"{f:.0f}_HX_fault"]["meta"]
        thresholds_by_freq[f] = departure_thresholds(meta, [f])
    print("\nthresholds from the interface-quantisation floor of each tone's own grid:")
    for f in args.freqs:
        t = thresholds_by_freq[f]
        print(f"  {f:7.0f} Hz  "
              f"dx={float(results[f'{f:.0f}_HX_fault']['meta']['dx_model_target_m']):.3f} m"
              f"   Hx {100*float(np.min(t['Cxx'])):.3f} %"
              f"   Hz {100*float(np.min(t['Cxz'])):.3f} %")

    print("\n=== fault model vs laterally invariant reference, same survey and grid ===")
    for f in args.freqs:
        def gains(src, tag):
            k = f"{f:.0f}_{src}_{tag}"
            return per_tx_gains(Path(results[k]["dir"]), results[k]["meta"])
        gxf, gxr = gains("HX", "fault"), gains("HX", "ref")
        gzf, gzr = gains("HZ", "fault"), gains("HZ", "ref")
        tx = gxf["tx_x"]; dist = FAULT_X_M - tx
        Cf = {"Cxx": gxf["hx"], "Cxz": gxf["hz"], "Czx": gzf["hx"], "Czz": gzf["hz"]}
        Cr = {"Cxx": gxr["hx"], "Cxz": gxr["hz"], "Czx": gzr["hx"], "Czz": gzr["hz"]}
        Cf["S"] = Cf["Cxz"] + Cf["Czx"]
        Cr["S"] = Cr["Cxz"] + Cr["Czx"]
        print(f"\n  {f:.0f} Hz")
        print(f"    {'observable':>10} {'threshold':>10} {'detect at':>11} "
              f"{'max departure':>14} {'ref |S|/|Cxx|':>14}")
        for name in ("Cxx", "Cxz", "Czx", "Czz", "S"):
            a = Cf[name][0, :, 0]
            b = Cr[name][0, :, 0]
            # `S` is a null on the reference, so normalise it by a real coupling
            # rather than by its own (near-zero) reference value.
            denom = (np.abs(Cr["Cxx"][0, :, 0]) if name == "S" else np.abs(b))
            dep = np.abs(a - b) / np.maximum(denom, 1e-300)
            thr = float(np.min(thresholds_by_freq[f][name]))
            ok = (dep > thr) & (dist > 0)
            det = float(np.max(dist[ok])) if np.any(ok) else None
            ref_null = float(np.median(np.abs(Cr["S"][0, :, 0])
                                       / np.maximum(np.abs(Cr["Cxx"][0, :, 0]), 1e-300)))
            print(f"    {name:>10} {100*thr:9.3f}% "
                  f"{(f'{det:9.1f} m' if det is not None else '        -'):>11} "
                  f"{100*np.max(dep):13.3f}% {ref_null:14.3e}")
            out["detection"].setdefault(f"{f:.0f}", {})[name] = det
            out["observables"].setdefault(f"{f:.0f}", {})[name] = {
                "dist_m": dist.tolist(), "departure": dep.tolist(),
                "threshold": thr}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    try:
        print("wrote", plot_profiles(out, Path(args.figure)))
    except Exception as exc:                       # plotting must never lose the data
        print(f"(figure not written: {exc})")
    print(f"\nDetection distances are the largest tool-to-fault distance at which the")
    print("observable differs from the SAME configuration on a laterally invariant")
    print("model by more than the quantisation floor. They are bounded by the survey")
    print(f"({far:.0f} m) and by the aperture half-width ({half:.0f} m), whichever is smaller -")
    print("a value at either limit is a lower bound, not a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
