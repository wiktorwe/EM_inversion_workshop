"""Does the TRUE model fit the data? The workshop's strongest self-check.

This is a synthetic study, so the answer is knowable: forward-model the model
the FDTD actually ran, calibrate it exactly as the inversion would, and compare
against the observed channel gains. If the true model does not fit, no inversion
result from that workspace means anything - it is what exposed a phase error of
up to 21.6 deg that every calibration-side check had missed.

It also answers the question interface snapping poses. The FD model's interfaces
are quantised onto each dataset's own grid; the analytic forward fits continuous
depths. `--compare-snapping` runs the check twice, with and without snapping
candidate interfaces onto each frequency's grid, and reports both reduced
chi-squareds - a measurement, not an argument.

    python scripts/experiments/true_model_check.py
    python scripts/experiments/true_model_check.py --compare-snapping
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.analytic_1d_forward import ForwardRejected  # noqa: E402
from scripts.modules.headless import matrix_setup  # noqa: E402
from scripts.modules.inversion_1d import (  # noqa: E402
    TENSOR_COMPONENTS,
    blocky_layers_from_trace,
    forward_tensor_for_tx,
    load_tensor_features,
    analytic_tensor_calibration,
    tensor_calibration,
)
from scripts.modules.segy import read_resistivity_from_segy  # noqa: E402


def true_params_for_tx(segy_path, tx_x, tx_z, z_start_rel, z_end_rel):
    """The TRUE Earth as a parameter vector in the INVERSION's parameterisation.

    Read from the SEG-Y, so these are the model's own CONTINUOUS interface
    depths - not one FD dataset's quantised copy of them. That is the point: the
    inversion fits continuous depths too, and the question is whether the true
    continuous Earth reproduces the data each quantised FD model actually
    produced.

    Built exactly as `unpack_model_params` builds a candidate: thicknesses fill
    the window `[z_start_rel, z_end_rel]`, so the deepest interior interface is
    the pinned window edge and every interface above it is a fitted one. That
    makes the comparison below a statement about the objective the inversion
    really minimises, not about a differently-shaped stack.
    """
    seg = read_resistivity_from_segy(str(segy_path))
    z = np.asarray(seg["z"], float)
    ix = int(np.argmin(np.abs(np.asarray(seg["x"], float) - float(tx_x))))
    d_abs, res = blocky_layers_from_trace(np.asarray(seg["resistivity"], float)[:, ix],
                                          z0=float(z[0]), dz=float(z[1] - z[0]))
    d_rel = np.asarray(d_abs, float) - float(tx_z)
    keep = (d_rel > float(z_start_rel)) & (d_rel < float(z_end_rel))
    # layer i is the one ABOVE interface i; keep the layers those interfaces bound
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise ValueError("no true interface falls inside the depth window")
    rho = np.concatenate([np.asarray(res, float)[idx], [float(res[idx[-1] + 1])]])
    d_in = np.concatenate([d_rel[idx], [float(z_end_rel)]])
    rho = np.concatenate([rho, [rho[-1]]])
    thk = np.diff(np.concatenate([[float(z_start_rel)], d_in]))
    if np.any(thk <= 0.0):
        raise ValueError(f"degenerate true stack for this Tx: thicknesses {thk}")
    n_layers = rho.size
    return np.concatenate([np.log10(rho), np.log10(thk)]), n_layers


def chi2_for_tx(tx_entry, segy_path, eps_r, cal, components, window, snap=None):
    """Reduced chi-squared of the TRUE model against this Tx's observed gains.

    `cal` may be a mapping or a callable taking `tx_entry`. The analytic budget
    is built from the transmitter's own data, so it cannot be shared across Tx
    the way a fitted calibration can.
    """
    z_start_rel, z_end_rel = window
    if callable(cal):
        cal = cal(tx_entry)
    params, n_layers = true_params_for_tx(
        segy_path, float(tx_entry.get("tx_x", 0.0)), float(tx_entry["tx_z"]),
        z_start_rel, z_end_rel)
    snap_dz, snap_org = (None, None) if snap is None else snap
    pred = forward_tensor_for_tx(params, tx_entry, n_layers, z_start_rel, z_end_rel,
                                 eps_r, components=components,
                                 snap_dz=snap_dz, snap_origin_m=snap_org)

    chi2, ndata = 0.0, 0
    per_comp = {}
    for c in components:
        obs = np.asarray(tx_entry["obs"][c], dtype=complex)
        sig = np.asarray(cal["sigma"][c], dtype=float)
        if sig.ndim == 1:                       # per frequency -> broadcast over rx
            sig = sig[:, None]
        C = np.asarray(cal["C"][TENSOR_COMPONENTS[c][0]], dtype=complex)[:, None]
        r = (C * pred[c] - obs) / sig
        s = float(np.sum(r.real ** 2 + r.imag ** 2))
        n = int(2 * r.size)
        per_comp[c] = s / max(n, 1)
        chi2 += s
        ndata += n
    return chi2 / max(ndata, 1), per_comp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-dir", default="workspace/2D/forward")
    ap.add_argument("--segy", default="examples/Fault_1.sgy")
    ap.add_argument("--z-start-rel", type=float, default=-60.0)
    ap.add_argument("--z-end-rel", type=float, default=60.0)
    ap.add_argument("--n-tx", type=int, default=5, help="How many Tx to check.")
    ap.add_argument("--compare-snapping", action="store_true")
    ap.add_argument("--calibration", choices=("analytic", "fitted"), default="analytic",
                    help="analytic = the computed C and error budget the inversion "
                         "uses; fitted = the FDTD calibration, for A/B only.")
    ap.add_argument("--out", default="workspace/2D/true_model_check.json")
    args = ap.parse_args()

    m = matrix_setup(Path(args.forward_dir))
    feats = load_tensor_features({s: v for s, v in m["by_source"].items()})
    components = tuple(feats["components"])
    if args.calibration == "analytic":
        order = json.loads(
            Path(m["meta_paths"][sorted(m["meta_paths"])[0]][0]).read_text())["fd_order"]
        cfg = {"dx": m["dx"], "fd_order": int(order), "eps_r": m["eps_r"],
               "components": components}
        cal = lambda tx_entry: analytic_tensor_calibration(  # noqa: E731
            cfg, tx_entry, components=components)
    else:
        cal = tensor_calibration(
            {s: [str(p) for p in v] for s, v in m["meta_paths"].items()})
    window = (args.z_start_rel, args.z_end_rel)

    tx_ids = sorted(feats["tx_data"])[: max(1, args.n_tx)]
    print(f"components : {', '.join(components)}")
    print(f"frequencies: {', '.join(f'{f:g}' for f in m['freqs'])} Hz")
    print(f"eps_r      : {', '.join(f'{e:.1f}' for e in m['eps_r'])}")
    print(f"calibration: {args.calibration}")
    print(f"true model read from {args.segy} (continuous depths)")
    print(f"depth window: {args.z_start_rel:g} .. {args.z_end_rel:g} m relative to Tx\n")

    modes = [("no snapping", None)]
    if args.compare_snapping:
        modes.append(("snapped per frequency", (m["dx"], m["snap_origin_m"])))

    report = {"components": list(components), "freqs_hz": m["freqs"].tolist(), "tx": {}}
    for label, snap in modes:
        vals, per_c = [], {}
        for tx_id in tx_ids:
            try:
                c2, pc = chi2_for_tx(feats["tx_data"][tx_id], args.segy, m["eps_r"],
                                     cal, components, window, snap=snap)
            except (ForwardRejected, ValueError) as exc:
                print(f"  Tx {tx_id}: skipped ({exc})")
                continue
            vals.append(c2)
            for k, v in pc.items():
                per_c.setdefault(k, []).append(v)
            report["tx"].setdefault(str(tx_id), {})[label] = c2
        if not vals:
            print(f"{label}: no Tx evaluated.")
            continue
        print(f"{label}: reduced chi2 over {len(vals)} Tx = "
              f"{np.mean(vals):.4f} (min {np.min(vals):.4f}, max {np.max(vals):.4f})")
        for k in components:
            print(f"      {k}: {np.mean(per_c[k]):.4f}")
        report.setdefault("summary", {})[label] = {
            "mean": float(np.mean(vals)), "min": float(np.min(vals)),
            "max": float(np.max(vals)), "n_tx": len(vals),
            "per_component": {k: float(np.mean(v)) for k, v in per_c.items()},
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
