"""Is the empymod line-source fallback the right physics for BOTH sources?

The `_EMPY_LINE_SIGN = -1` constant in `scripts/modules/empymod_line_source.py`
was derived from source-polarity conventions (rockem `+K` vs Hunziker `-J^m`)
and then measured, for the Kx source only. When Kz support was added, the same
question reopened for `ab=46`/`ab=66`: a wrong `ab` code is not a subtle error,
it selects a different Green's function entirely.

The check is the ratio empymod / native, COMPLEX. Amplitude alone cannot tell a
sign convention from a time convention - the mistake that has already cost time
on this codebase - so the conjugate ratio is reported alongside.

The default quadrature in `empymod_line_yintegral` (uniform, n_y=120) is NOT
accurate enough for this: it leaves up to 14 % at near offsets. This script
builds its own GRADED y-grid, dense near y=0 where the integrand peaks, which is
what the original Kx measurement used.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from scripts.modules.analytic_1d_forward import Layer1D, forward_1d_gains  # noqa: E402
from scripts.modules.empymod_line_source import (  # noqa: E402
    _AB_FOR_SOURCE, _EMPY_LINE_SIGN, _require_empymod,
)
from scripts.modules.rockem_bridge import model  # noqa: E402


def graded_y(y_max_m: float, n_half: int) -> np.ndarray:
    """Symmetric y nodes, geometrically graded from ~1 cm out to `y_max_m`."""
    half = np.geomspace(0.01, float(y_max_m), int(n_half))
    return np.concatenate([-half[::-1], [0.0], half])


def empymod_line(off_x, freq_hz, layers, tx_z, rx_z, ab, y):
    empymod = _require_empymod()
    depth, res, epermH = model.layers_to_stack(layers, tx_z)
    out = np.zeros(np.size(off_x), dtype=complex)
    for i, off in enumerate(np.atleast_1d(off_x)):
        resp = empymod.dipole(
            src=[0.0, y, float(tx_z)], rec=[float(off), 0.0, float(rx_z)],
            depth=depth, res=res, freqtime=np.asarray([freq_hz], dtype=float),
            ab=int(ab), epermH=epermH, epermV=epermH,
            aniso=np.ones(len(res)), verb=0,
        )
        out[i] = _EMPY_LINE_SIGN * np.trapezoid(np.asarray(resp, dtype=complex).ravel(), y)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y-max", type=float, default=60_000.0)
    ap.add_argument("--n-half", type=int, default=3000)
    ap.add_argument("--tol", type=float, default=5e-3,
                    help="max |ratio - 1| accepted, after the -1 sign is applied")
    args = ap.parse_args()

    # Same 3-layer stack, geometry and tones as the original Kx derivation.
    rho = np.array([30.0, 5.0, 80.0])
    thk = np.array([40.0, 25.0])
    eps_r = 7.0
    tx_z, rx_z = 100.0, 115.0
    offs = np.array([-13.1, 13.1, -25.3, 25.3])
    freqs = np.array([1000.0, 4000.0])

    layers = [Layer1D(rho[0], thk[0], eps_r), Layer1D(rho[1], thk[1], eps_r),
              Layer1D(rho[2], None, eps_r)]
    y = graded_y(args.y_max, args.n_half)
    print(f"quadrature: graded, y_max={args.y_max:.0f} m, {y.size} nodes\n")

    print(f"{'src':>4} {'recv':>5} {'ab':>4} {'f Hz':>7} {'off m':>7} "
          f"{'ratio (empymod/native)':>28} {'|conj ratio - 1|':>17}")
    worst = 0.0
    for src, (ab_hx, ab_hz) in sorted(_AB_FOR_SOURCE.items()):
        for f in freqs:
            hx_n, hz_n = forward_1d_gains(rho, thk, np.asarray([f]), offs, tx_z, rx_z,
                                          eps_r, source_field=src)
            for recv, ab, native in (("Hx", ab_hx, hx_n[0]), ("Hz", ab_hz, hz_n[0])):
                emp = empymod_line(offs, float(f), layers, tx_z, rx_z, ab, y)
                for k, off in enumerate(offs):
                    r = emp[k] / native[k]
                    dev = abs(r - 1.0)
                    worst = max(worst, dev)
                    print(f"{src:>4} {recv:>5} {ab:>4} {f:7.0f} {off:7.1f} "
                          f"{r.real:+13.6f}{r.imag:+12.6f}j {abs(np.conj(r) - 1.0):17.6f}")
    print(f"\nworst |ratio - 1| over every source, receiver, tone and offset: {worst:.2e}")

    # CONTROL. Every ratio above comes out at the same 1.000064, which is what a
    # converged comparison SHOULD look like (the leftover is a common quadrature
    # bias) but is also what a harness that compares something against itself
    # would look like. So pair a native response with the WRONG ab deliberately:
    # if that also passes, the check above proves nothing.
    hx_x, _ = forward_1d_gains(rho, thk, np.asarray([freqs[0]]), offs, tx_z, rx_z,
                               eps_r, source_field="HX")
    wrong = empymod_line(offs, float(freqs[0]), layers, tx_z, rx_z, 46, y) / hx_x[0]
    wrong_dev = float(np.max(np.abs(wrong - 1.0)))
    print(f"control (Cxx native vs ab=46, a Kz source): ratio {wrong[0]:.4f}, "
          f"|ratio - 1| = {wrong_dev:.3g}")

    ok = worst <= args.tol and wrong_dev > 0.1
    if worst > args.tol:
        print(f"VERDICT: FAIL - deviation exceeds {args.tol:.1e}")
    elif wrong_dev <= 0.1:
        print("VERDICT: FAIL - the CONTROL passed too, so this harness is not "
              "actually comparing independent quantities. Fix the test first.")
    else:
        print("VERDICT: PASS - the -1 convention and the ab codes hold for Kx AND "
              "Kz, and a wrong ab is rejected")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
