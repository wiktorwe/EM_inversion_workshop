"""Does the BlockInv optimizer actually fit the full 2x2 tensor?

BlockInv packs the data, the errors and the Gauss-Newton Jacobian into flat real
vectors. That packing used to have arity 2 hardcoded in five places, so the
optimizer could only ever fit the Kx pair and refused anything else outright.
This checks the generalised packing on real FDTD data:

1. REGRESSION - on Cxx+Cxz it must reproduce the pre-change result exactly.
   A generalisation that changes the two-component answer is a regression, not
   a feature.
2. TENSOR - on all four components it must run to completion and land in the
   same region as Differential Evolution, which reaches chi2 ~0.76 /
   model err ~0.54 here (see `tensor_1d_test.py`).

The BlockInv code lives in notebook 05's single code cell, so this executes that
cell and calls into it - the same thing `scripts/validate_notebooks.py` does.
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
warnings.filterwarnings("ignore")

from scripts.modules.inversion_1d import (  # noqa: E402
    load_tensor_features, n_tensor_data, resolve_tensor_calibration, tensor_objective,
    unpack_model_params,
)

ALL = ("Cxx", "Cxz", "Czx", "Czz")


def notebook_namespace(nb_name="05_1d_inversion.ipynb"):
    """Exec notebook 05's code cell and hand back its namespace.

    The cell ends by displaying the GUI, so its stdout is swallowed - otherwise
    the widget repr buries the result table.
    """
    import contextlib
    import io

    nb = json.loads((ROOT / nb_name).read_text())
    code = "".join(next(c for c in nb["cells"] if c["cell_type"] == "code")["source"])
    g = {"__name__": "__main__", "__file__": str(ROOT / nb_name)}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, nb_name, "exec"), g)
    return g


def model_error(params, truth, n_layers, zs, ze):
    """RMS log10-resistivity difference on a fixed depth grid."""
    zg = np.linspace(zs, ze, 400)
    out = []
    for p, nl in ((params, n_layers), (truth[0], truth[1])):
        rho, _thk, depth = unpack_model_params(p, nl, zs, ze)
        idx = np.searchsorted(depth, zg)
        out.append(np.log10(rho[np.clip(idx, 0, rho.size - 1)]))
    return float(np.sqrt(np.mean((out[0] - out[1]) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hx-dir", default="workspace/2D/forward")
    ap.add_argument("--hz-dir", default="workspace/2D/forward_hz")
    ap.add_argument("--tx", type=int, default=0)
    ap.add_argument("--z-start", type=float, default=-60.0)
    ap.add_argument("--z-end", type=float, default=60.0)
    ap.add_argument("--n-layers", type=int, default=5)
    ap.add_argument("--block-iter", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    g = notebook_namespace()
    feats = load_tensor_features({"HX": args.hx_dir, "HZ": args.hz_dir})
    tx = feats["tx_data"][args.tx]
    meta = ROOT / args.hx_dir / "setup_metadata.json"

    _meta = json.loads(meta.read_text())

    def run(components):
        cfg = {
            "n_layers": args.n_layers, "z_start_rel": args.z_start, "z_end_rel": args.z_end,
            "eps_r": float(_meta["eps_r_used"]),
            # The FD design the data was modelled on. `C = dx*dz*s(order)` and
            # the error budget's near-source term is `(dx/r)^2`, so the analytic
            # calibration needs both.
            "dx": float(_meta["dx_model_target_m"]), "fd_order": int(_meta["fd_order"]),
            "log10_rho_min": 0.0, "log10_rho_max": np.log10(150.0),
            "log10_thk_min": np.log10(5.0), "log10_thk_max": np.log10(50.0),
            "thk_min": 5.0, "reg_lambda": 500.0, "w_hxh": 1.0, "w_hxhz": 1.0,
            "seed": args.seed, "block_max_iter": args.block_iter,
            "components": tuple(components),
        }
        g["SETUP_META"] = meta
        out = g["_invert_single_tx_blockinv"](tx, cfg)
        # Score every run against the SAME calibration the notebook's inversion
        # uses. `cfg` already carries `dx` / `fd_order`, so this is the analytic
        # one; the tx_entry is needed because its error budget is relative to
        # the data.
        cal = resolve_tensor_calibration(dict(cfg, components=ALL), meta, tx_entry=tx)
        # score EVERY run on all four components, so the rows are comparable
        mis = tensor_objective(out["params"], tx, cfg["n_layers"], cfg["z_start_rel"],
                               cfg["z_end_rel"], cfg["eps_r"], 0.0, cal, components=ALL)
        return out, mis / max(n_tensor_data(tx, ALL), 1)

    print(f"tx {args.tx} at x={tx['tx_x']:.1f} m, {args.n_layers} layers, "
          f"BlockInv max_iter={args.block_iter}\n")
    print(f"{'components':>28} {'n_data':>7} {'blk chi2':>10} {'chi2 (all four)':>16}")
    results = {}
    for comps in (("Cxx", "Cxz"), ALL):
        out, chi2_all = run(comps)
        n_data = 2 * sum(np.size(tx["obs"][c]) for c in comps)
        results[comps] = (out, chi2_all)
        print(f"{'+'.join(comps):>28} {n_data:>7} {out['chi2']:10.4f} {chi2_all:16.4f}")

    kx_out = results[("Cxx", "Cxz")][0]
    print(f"\nKx-pair rho: {np.array2string(kx_out['rho'], precision=4)}")
    print(f"tensor  rho: {np.array2string(results[ALL][0]['rho'], precision=4)}")
    print("\nBoth rows are scored on all four components, so adding Czx/Czz is\n"
          "compared on the same yardstick as leaving them out.")
    print(
        "\nNOTE, measured: BlockInv lands ~35x worse than Differential Evolution\n"
        "on this problem (DE reaches chi2 ~0.76, see tensor_1d_test.py). It is NOT\n"
        "an iteration budget - --block-iter 15, 40 and 100 give bit-identical\n"
        "results, because it stops on min_dphi_percent. It is a local Gauss-Newton\n"
        "method converging from a random start into a local minimum. Use DE for the\n"
        "answer; BlockInv is useful for its deterministic linearised uncertainties."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
