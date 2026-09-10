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


def _code_cells(nb_path: Path) -> str:
    """EVERY code cell, concatenated, in order.

    Voila runs all of them, so validation must too. Notebook 04 has two - a
    setup cell and a 959-line GUI cell - so taking only the first would leave
    more than half of it unchecked, and a NameError in the GUI cell would
    surface only in front of a user.
    """
    nb = json.loads(nb_path.read_text())
    cells = ["".join(c.get("source", []))
             for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if not cells:
        raise RuntimeError(f"No code cell in {nb_path.name}")
    return "\n\n".join(cells)


def _run_cell(code: str, nb_name: str) -> tuple[bool, str]:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    g: dict = {"__name__": "__main__", "__file__": str(ROOT / nb_name)}
    # A live kernel injects `display` into the user namespace, so notebook cells
    # legitimately use it without importing it. Provide it here for the same
    # reason - otherwise every such cell fails validation for a reason that
    # cannot happen in Voila.
    try:
        from IPython.display import display as _display
        g["display"] = _display
    except ImportError:
        g["display"] = lambda *a, **k: None
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
        code = _code_cells(nb_path)
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
