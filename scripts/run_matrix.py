#!/usr/bin/env python3
"""Model and calibrate EVERY dataset of the acquisition matrix, in one command.

Step 01 already recorded which frequencies and which source components were
selected. Nothing downstream should make you re-enter that choice one dataset at
a time: going from one broadband run to N frequencies x M sources should cost
less wall-clock time, not more clicking.

    python scripts/run_matrix.py                      # model + calibrate everything
    python scripts/run_matrix.py --skip-modelling     # datasets already modelled
    python scripts/run_matrix.py --method homogeneous_rho_min
    python scripts/run_matrix.py --dry-run            # show the plan, run nothing

Both stages are sequential and per-dataset, and a failure in one dataset is
reported and stepped over rather than abandoning the batch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modules.workshop_config import load_config  # noqa: E402


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None, help="Workshop root.")
    ap.add_argument("--nproc", type=int, default=None,
                    help="MPI ranks per job (default: the configured nproc).")
    ap.add_argument("--method", default="lateral_average_true",
                    choices=["lateral_average_true", "homogeneous_rho_min"],
                    help="Calibration Earth model, applied to EVERY dataset. "
                         "One method for all of them: mixing them across sources "
                         "is what the tensor inversion refuses.")
    ap.add_argument("--skip-modelling", action="store_true",
                    help="Only calibrate; assumes Data/ already exists per dataset.")
    ap.add_argument("--skip-calibration", action="store_true",
                    help="Only run the forward modelling.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would run, then exit.")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    root = (args.root or ROOT).expanduser().resolve()
    cfg = load_config(root)
    nproc = int(args.nproc or cfg.nproc_default)

    from scripts.modules.headless import iter_datasets, run_calibration_matrix, run_forward

    fwd_root = cfg.fwd_2d_dir
    datasets = iter_datasets(fwd_root)
    if not datasets:
        print(f"ERROR: no forward datasets under {fwd_root}. Run Step 01 first.",
              file=sys.stderr)
        return 1

    print(f"{len(datasets)} dataset(s) under {fwd_root}, -np {nproc}:")
    for d in datasets:
        print(f"   {d['name']:>16}  source={d['source_field']:<3} freq={d['freq_hz']}")
    print(f"\nplan: {'skip' if args.skip_modelling else 'run'} modelling, "
          f"{'skip' if args.skip_calibration else 'run'} calibration "
          f"({args.method}, each dataset with its own source)")
    if args.dry_run:
        return 0

    t0 = time.time()
    failures = []

    if not args.skip_modelling:
        for i, d in enumerate(datasets, start=1):
            print(f"\n[{i}/{len(datasets)}] modelling {d['name']}")
            try:
                run_forward(d["run_dir"], nproc=nproc)
            except Exception as exc:                      # noqa: BLE001
                failures.append(f"modelling {d['name']}: {type(exc).__name__}: {exc}")
                print(f"       FAILED: {failures[-1]}")

    if not args.skip_calibration:
        results = run_calibration_matrix(fwd_root, method=args.method, nproc=nproc,
                                         datasets=datasets)
        failures += [f"calibration {r['name']} [{r['source_field']}]: {r['error']}"
                     for r in results if not r.get("ok")]

    dt = time.time() - t0
    print(f"\nTotal wall time: {dt:.1f} s ({dt / 60.0:.1f} min)")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All datasets modelled and calibrated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
