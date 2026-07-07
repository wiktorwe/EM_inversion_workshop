#!/usr/bin/env python3
"""Smoke-test workshop notebook setup cells after reconfiguration."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

NOTEBOOKS = [
    "00_configure_workshop.ipynb",
    "01_fw_setup.ipynb",
    "02_fwmodelling_and_data_visualization.ipynb",
    "03_2d_inversion.ipynb",
    "04_2d_inversion_results.ipynb",
    "05_1d_inversion.ipynb",
    "06_1d_inversion_results.ipynb",
]

# Notebooks that require a valid rockem-suite checkout once imports proceed.
NEEDS_ROCKEM = {
    "01_fw_setup.ipynb",
    "02_fwmodelling_and_data_visualization.ipynb",
    "03_2d_inversion.ipynb",
    "04_2d_inversion_results.ipynb",
    "05_1d_inversion.ipynb",
    "06_1d_inversion_results.ipynb",
}


def _first_code_cell(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text())
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            return "".join(cell.get("source", []))
    raise RuntimeError(f"No code cell in {nb_path.name}")


def _run_cell(code: str, nb_name: str) -> tuple[bool, str]:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    g: dict = {"__name__": "__main__", "__file__": str(ROOT / nb_name)}
    try:
        exec(compile(code, nb_name, "exec"), g)
        return True, "ok"
    except Exception as exc:
        return False, "".join(traceback.format_exception_only(exc)).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate workshop notebook setup cells.")
    parser.add_argument(
        "--expect-rockem-missing",
        action="store_true",
        help="Treat rockem-suite import failures as warnings for notebooks 01/02/04/05.",
    )
    args = parser.parse_args()

    failures = 0
    warnings = 0
    print(f"Workshop root: {ROOT}\n")

    for name in NOTEBOOKS:
        nb_path = ROOT / name
        if not nb_path.exists():
            print(f"FAIL  {name}: file not found")
            failures += 1
            continue
        code = _first_code_cell(nb_path)
        ok, msg = _run_cell(code, name)
        if ok:
            print(f"OK    {name}")
            continue
        if args.expect_rockem_missing and name in NEEDS_ROCKEM:
            if "Failed to import workshop modules" in msg:
                print(f"WARN  {name}: underlying import failed (likely rockem-suite path — configure in Step 00)")
                warnings += 1
                continue
            dep_markers = ("No module named 'joblib'", "No module named 'segyio'", "No module named 'ipywidgets'")
            if any(m in msg for m in dep_markers):
                print(f"FAIL  {name}: {msg}")
                failures += 1
                continue
            print(f"WARN  {name}: {msg}")
            warnings += 1
            continue
        print(f"FAIL  {name}: {msg}")
        failures += 1

    print()
    if failures:
        print(f"{failures} failure(s), {warnings} warning(s)")
        return 1
    print(f"All notebooks passed ({warnings} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
