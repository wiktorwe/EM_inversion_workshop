"""Quantify the artificial-permittivity bias the calibration cannot see.

`design_explicit_fd` inflates `eps_r` (up to `eps_r_cap`, subject to a loss-
tangent floor) purely to buy a larger explicit time step, and
`analytic_1d_forward` deliberately evaluates the analytic reference at that SAME
`eps_r_used`. That is the right thing for the calibration - it isolates genuine
discretisation error - but it means the inflation CANCELS out of `C(f)` and is
invisible there: FDTD and analytic can agree perfectly while both differ from
the true physics.

This script measures the size of that offset directly, by evaluating the
analytic solver twice on the same Earth model: once at `eps_r_used` and once at
a physical `eps_r`. Nothing here involves the FDTD engine - it runs in seconds.

Usage:
    python scripts/experiments/eps_r_bias.py [--forward-dir workspace/2D/forward]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.analytic_1d_forward import Layer1D  # noqa: E402
from scripts.modules.fdtd_analytic_calibration import (  # noqa: E402
    calibration_geometry,
    calibration_geometry_production,
    homogeneous_analytic_gains,
    layered_analytic_gains,
    load_setup_metadata,
)

EPS0 = 8.854187817e-12


def _loss_tangent(rho, f_hz, eps_r):
    return (1.0 / rho) / (2 * np.pi * f_hz * eps_r * EPS0)


def _report(name, freqs, hx_a, hz_a, hx_b, hz_b, eps_a, eps_b):
    print(f"\n=== {name}: eps_r={eps_a:.1f} (used by the FD design) vs eps_r={eps_b:g} (physical) ===")
    print(f"{'f [Hz]':>8} {'Hx |d| %':>10} {'Hx dphase':>11} {'Hz |d| %':>10} {'Hz dphase':>11}")
    out = []
    for i, f in enumerate(freqs):
        da = 100.0 * np.max(np.abs(np.abs(hx_a[i]) / np.abs(hx_b[i]) - 1.0))
        dp = np.max(np.abs(np.angle(hx_a[i] / hx_b[i], deg=True)))
        dz = 100.0 * np.max(np.abs(np.abs(hz_a[i]) / np.abs(hz_b[i]) - 1.0))
        dpz = np.max(np.abs(np.angle(hz_a[i] / hz_b[i], deg=True)))
        print(f"{f:8.0f} {da:10.4f} {dp:+10.4f}° {dz:10.4f} {dpz:+10.4f}°")
        out.append({"f_hz": float(f), "hx_amp_pct": float(da), "hx_phase_deg": float(dp),
                    "hz_amp_pct": float(dz), "hz_phase_deg": float(dpz)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-dir", default="workspace/2D/forward")
    ap.add_argument("--eps-physical", type=float, default=7.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fwd = Path(args.forward_dir)
    meta = load_setup_metadata(fwd / "setup_metadata.json")
    freqs = np.asarray(meta["flist_hz"], dtype=float)
    eps_used = float(meta["eps_r_used"])
    eps_phys = float(args.eps_physical)
    results = {"eps_r_used": eps_used, "eps_r_physical": eps_phys}

    # --- homogeneous rho_min, the calibration's own homogeneous geometry ----
    geo = calibration_geometry(meta)
    rho_min = float(meta["rho_min_ohm_m"])
    hx_a, hz_a = homogeneous_analytic_gains(freqs, geo["off_x"], geo["tx_z"], geo["rx_z"], rho_min, eps_used)
    hx_b, hz_b = homogeneous_analytic_gains(freqs, geo["off_x"], geo["tx_z"], geo["rx_z"], rho_min, eps_phys)
    results["homogeneous"] = _report(
        f"homogeneous rho={rho_min:g} Ohm-m", freqs, hx_a, hz_a, hx_b, hz_b, eps_used, eps_phys)
    print(f"  loss tangent at f_max, rho={rho_min:g}: "
          f"{_loss_tangent(rho_min, freqs.max(), eps_used):.0f} (used) / "
          f"{_loss_tangent(rho_min, freqs.max(), eps_phys):.0f} (physical)")

    # --- lateral-average layered Earth, the production calibration model ----
    layers_path = fwd / "calibration_lateral_average" / "lateral_average_layers.json"
    if layers_path.exists():
        payload = json.loads(layers_path.read_text())
        geo_p = calibration_geometry_production(meta)

        def stack(eps):
            return [Layer1D(float(L["resistivity_ohm_m"]),
                            None if L.get("thickness_m") is None else float(L["thickness_m"]),
                            float(eps)) for L in payload["layers"]]

        hx_a, hz_a = layered_analytic_gains(freqs, geo_p["off_x"], geo_p["tx_z"], geo_p["rx_z"], stack(eps_used))
        hx_b, hz_b = layered_analytic_gains(freqs, geo_p["off_x"], geo_p["tx_z"], geo_p["rx_z"], stack(eps_phys))
        rho_mean = float(np.mean([L["resistivity_ohm_m"] for L in payload["layers"]]))
        results["lateral_average"] = _report(
            f"lateral-average layered (mean rho={rho_mean:.1f} Ohm-m, production offsets)",
            freqs, hx_a, hz_a, hx_b, hz_b, eps_used, eps_phys)
        rho_max = float(meta["rho_max_ohm_m"])
        print(f"  loss tangent at f_max, rho_max={rho_max:g} (the worst corner): "
              f"{_loss_tangent(rho_max, freqs.max(), eps_used):.0f} (used) / "
              f"{_loss_tangent(rho_max, freqs.max(), eps_phys):.0f} (physical)")
    else:
        print(f"\n(skipping lateral-average: {layers_path} not found)")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
