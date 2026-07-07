#!/usr/bin/env python3
"""Add nbformat cell ids and normalize workshop notebooks."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import nbformat
from nbformat.validator import normalize


def normalize_notebook(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    normalize(nb)
    for cell in nb.cells:
        if not cell.get("id"):
            cell["id"] = uuid.uuid4().hex[:8]
    nb.nbformat_minor = max(int(getattr(nb, "nbformat_minor", 0) or 0), 5)
    nbformat.write(nb, path)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    notebooks = sorted(root.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 1
    for path in notebooks:
        normalize_notebook(path)
        print(f"normalized {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
