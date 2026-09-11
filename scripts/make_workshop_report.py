#!/usr/bin/env python3
"""Generate a LaTeX workflow report from the current workshop workspace.

Run from the workshop root (or pass --root). Requires Step 01
``setup_metadata.json``. Writes one report for the whole acquisition matrix
and the selected 2D ladder / 1D run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modules.workshop_config import load_config  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write one LaTeX report of the acquisition matrix, modelled data, "
            "and any 2D/1D inversion results under workspace/."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Workshop root (default: directory containing this script's parent).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Run pdflatex twice on the generated .tex (if pdflatex is on PATH).",
    )
    parser.add_argument(
        "--2d-run",
        dest="run_2d",
        default=None,
        help="2D inversion ladder to include (RunN, N, or a directory). Default: latest.",
    )
    parser.add_argument(
        "--1d-run",
        dest="run_1d",
        default=None,
        help="1D inversion run to include (RunN, N, or a directory). Default: latest.",
    )
    parser.add_argument(
        "--no-2d",
        action="store_true",
        help="Omit the 2D inversion section even if a run exists.",
    )
    parser.add_argument(
        "--no-1d",
        action="store_true",
        help="Omit the 1D inversion section even if a run exists.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List the forward datasets in the acquisition matrix, then exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = (args.root or ROOT).expanduser().resolve()
    load_config(root)

    from scripts.modules.workshop_report import build_report, forward_datasets

    datasets = forward_datasets(root)
    if args.list_datasets:
        if not datasets:
            print("No forward datasets found - run Step 01 first.")
            return 1
        for d in datasets:
            print(f"{d['name']:>20}  source={d['source_field']:<3} "
                  f"freq={d['freq_hz']}  {d['run_dir']}")
        return 0

    try:
        result = build_report(
            root=root,
            include_2d=not args.no_2d,
            include_1d=not args.no_1d,
            run_2d=args.run_2d,
            run_1d=args.run_1d,
            compile_pdf_flag=bool(args.compile),
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    tex_path = result["tex_path"]
    pdf_path = result["pdf_path"]
    report_dir = result["report_dir"]
    print(f"Wrote {tex_path}")
    nfig = len(result["figures"])
    print(f"Wrote {nfig} figure(s) under {report_dir / 'figures'}")
    n_ds = result.get("n_datasets") or 0
    if n_ds:
        print(f"Acquisition matrix: {n_ds} dataset(s).")
    if result["run_2d"] is not None:
        print(f"Included 2D inversion: {Path(result['run_2d']).name}")
    else:
        print("No 2D inversion run included.")
    if result["run_1d"] is not None:
        print(f"Included 1D inversion: {Path(result['run_1d']).name}")
    else:
        print("No 1D inversion run included.")
    for note in result["notes"]:
        print(f"Note: {note}")
    if pdf_path is not None:
        print(f"Compiled {pdf_path}")
        return 0
    if result.get("compile_error"):
        print(f"PDF compile failed: {result['compile_error']}", file=sys.stderr)
    if args.compile:
        print(
            "Compile from the report directory with:\n"
            f"  cd {report_dir} && pdflatex -interaction=nonstopmode {tex_path.name}",
            file=sys.stderr,
        )
        return 1
    print(
        "To compile the PDF:\n"
        f"  cd {report_dir} && pdflatex -interaction=nonstopmode {tex_path.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
