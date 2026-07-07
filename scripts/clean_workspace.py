#!/usr/bin/env python3
"""Remove all notebook-generated artifacts and restore a pristine workshop tree."""

from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _artifact_paths(root: Path, include_tmp: bool) -> list[Path]:
    paths: list[Path] = []

    workspace = root / "workspace"
    if workspace.exists():
        paths.append(workspace)

    # Legacy layout (pre-reorg) — safe to remove if still present
    for legacy in (
        root / "FDmodel",
        root / "InversionInput",
        root / "Results",
    ):
        if legacy.exists():
            paths.append(legacy)
    for legacy_run in root.glob("InversionRun*"):
        if legacy_run.is_dir():
            paths.append(legacy_run)

    for pid in root.glob(".voila*.pid"):
        paths.append(pid)

    for cache in root.rglob("__pycache__"):
        paths.append(cache)
    for checkpoint in root.rglob(".ipynb_checkpoints"):
        paths.append(checkpoint)
    for pyc in root.rglob("*.py[cod]"):
        paths.append(pyc)

    if include_tmp:
        for tmp in glob.glob("/tmp/em_workshop_gui_*"):
            paths.append(Path(tmp))

    return sorted(set(paths))


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove workshop-generated artifacts (workspace/, pids, caches).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting.",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    parser.add_argument(
        "--include-tmp",
        action="store_true",
        help="Also remove /tmp/em_workshop_gui_* session folders.",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    targets = _artifact_paths(root, include_tmp=args.include_tmp)
    if not targets:
        print("Nothing to clean — workshop is already pristine.")
        return 0

    print("The following paths will be removed:")
    for path in targets:
        print(f"  {_rel(root, path)}")

    if args.dry_run:
        print("\n(dry run — no files deleted)")
        return 0

    if not args.yes:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 1

    for path in targets:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists() or path.is_symlink():
            path.unlink(missing_ok=True)

    print(f"\nCleaned {len(targets)} path(s). Workshop restored to pristine state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
