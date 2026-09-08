"""Minimal, exact-match editor for the workshop's Voila notebooks.

The six workshop notebooks keep essentially all their code in one very large
cell, stored in ``.ipynb`` JSON as a list of newline-terminated strings. Editing
that with a text tool is unreliable (JSON escaping, line splits), so this helper
joins a cell's source, applies exact string replacements, and splits it back the
way nbformat expects.

Usage:
    python scripts/dev/nbedit.py <notebook.ipynb> <edits.json>

``edits.json`` is a list of ``{"old": ..., "new": ..., "count": n}`` objects
applied in order to whichever code cell contains ``old``. ``count`` defaults to
1 and is enforced: a replacement that does not fire exactly ``count`` times is
an error, so a stale edit fails loudly instead of silently doing nothing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def detect_indent(raw: str, default: int = 1) -> int:
    """The notebook's own JSON indentation, so an edit does not reformat the file.

    The workshop's notebooks are not consistent with each other (some are written
    with indent=1, some with indent=2); rewriting with the wrong one turns a
    two-line change into a whole-file diff and makes review impossible.
    """
    m = re.search(r'^\{\n( +)"', raw)
    return len(m.group(1)) if m else default


def split_source(text: str) -> list[str]:
    lines = text.split("\n")
    out = [ln + "\n" for ln in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def apply_edits(nb_path: Path, edits: list[dict]) -> None:
    raw = nb_path.read_text()
    nb = json.loads(raw)
    indent = detect_indent(raw)
    for i, edit in enumerate(edits):
        old, new = edit["old"], edit["new"]
        want = int(edit.get("count", 1))
        hits = 0
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            n = src.count(old)
            if n == 0:
                continue
            hits += n
            cell["source"] = split_source(src.replace(old, new))
        if hits != want:
            raise SystemExit(
                f"edit {i} in {nb_path.name}: expected {want} occurrence(s) of\n"
                f"---\n{old}\n---\nbut found {hits}"
            )
    nb_path.write_text(json.dumps(nb, indent=indent, ensure_ascii=False) + "\n")
    print(f"{nb_path.name}: applied {len(edits)} edit(s)")


if __name__ == "__main__":
    apply_edits(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()))
