#!/usr/bin/env python3
"""Smoke test for invert_tx_hybrid_population."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.modules.inversion_hybrid import HYBRID_OPTIMIZER_ID, invert_tx_hybrid_population


def main() -> int:
    nfreq, nrx = 2, 2
    tx = {
        "tx_id": 0,
        "tx_x": 0.0,
        "tx_z": 6000.0,
        "off_x": np.array([10.0, 20.0]),
        "off_z": np.zeros(2),
        "freqs": np.array([2000.0, 4000.0]),
    }
    tx["obs"] = {
        "Cxx": np.ones((nfreq, nrx), dtype=complex),
        "Cxz": np.ones((nfreq, nrx), dtype=complex) * 0.5,
    }
    cfg = {
        "n_layers": 3,
        "z_start_rel": -30.0,
        "z_end_rel": 30.0,
        "eps_r": 7.0,
        "log10_rho_min": 0.0,
        "log10_rho_max": 3.0,
        "log10_thk_min": np.log10(5.0),
        "log10_thk_max": np.log10(40.0),
        "reg_lambda": 100.0,
        "maxiter": 2,
        "popsize": 4,
        "block_max_iter": 2,
        "thk_min": 5.0,
        "thk_max": 40.0,
        "components": ("Cxx", "Cxz"),
        "seed": 0,
    }
    cal = {
        "C": {"HX": np.ones(nfreq, dtype=complex), "HZ": np.ones(nfreq, dtype=complex)},
        "sigma": {
            "Cxx": np.full(nfreq, 0.03),
            "Cxz": np.full(nfreq, 0.03),
        },
    }
    cfg["tensor_calibration"] = cal
    out = invert_tx_hybrid_population(tx, cfg, cal=cal, seed=0)
    assert out["optimizer"] == HYBRID_OPTIMIZER_ID
    assert out["envelope"]["z"].size > 0
    assert out["n_envelope"] > 0
    assert out["rho"].size == cfg["n_layers"]
    print("OK:", out["chi2"], "n_envelope=", out["n_envelope"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
