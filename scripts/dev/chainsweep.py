#!/usr/bin/env python3
"""Does every notebook resolve its dataset paths on an ACQUISITION MATRIX?

The bug this exists to prevent, in full:

    Step 01 builds one dataset per (frequency, source), each in its own
    subdirectory with its own setup_metadata.json. The forward ROOT then holds
    manifest.json and subdirectories - and NO dataset artifacts at all.

    Step 05 bound `SETUP_META = FWD_2D_DIR / "setup_metadata.json"` at import
    and never rebound it, so it read a file that cannot exist on a matrix
    workspace. Shipped, and only found by a user:
        "Lambda tune failed: [Errno 2] No such file or directory:
         .../workspace/2D/forward/setup_metadata.json"

`validate_notebooks.py` did not catch it because the repo's own workspace was a
single-dataset one, where the root path DOES exist. `bugsweep`/`handlersweep`
did not catch it because the name was defined and the handler only failed once
it touched the filesystem.

So this executes every notebook against a synthetic MATRIX workspace and asserts
the invariant that actually matters:

    NO module-level Path global may point at a dataset artifact
    sitting directly in the forward root.

`manifest.json` is the one thing that legitimately lives there.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Artifacts that belong to a DATASET, never to the forward root.
DATASET_ARTIFACTS = {
    "setup_metadata.json", "sg.rss", "ep.rss", "wav2d.rss", "Survey.rss",
    "mod.cfg", "Hxshot.rss", "Hzshot.rss", "runmod.sh", "mpiqueue.log",
}
NOTEBOOKS = ["02_fwmodelling_and_data_visualization.ipynb", "03_2d_inversion.ipynb",
             "04_2d_inversion_results.ipynb", "05_1d_inversion.ipynb",
             "06_1d_inversion_results.ipynb"]


def build_matrix_workspace(dst: Path, donor: Path, freqs=(2000, 4000), sources=("hx", "hz")):
    """A forward root holding ONLY manifest.json plus dataset subdirectories."""
    fwd = dst / "2D" / "forward"
    fwd.mkdir(parents=True, exist_ok=True)
    runs = {}
    meta = json.loads((donor / "setup_metadata.json").read_text())
    for f in freqs:
        for src in sources:
            name = f"f{f}Hz_{src}"
            d = fwd / name
            (d / "Data").mkdir(parents=True, exist_ok=True)
            m = dict(meta)
            m["flist_hz"] = [float(f)]
            m["f_min_hz"] = float(f)
            m["f_max_hz"] = float(f)
            m["source_field"] = src.upper()
            (d / "setup_metadata.json").write_text(json.dumps(m, indent=2))
            for fn in ("sg.rss", "ep.rss", "wav2d.rss", "Survey.rss", "mod.cfg"):
                srcf = donor / fn
                if srcf.exists():
                    shutil.copy(srcf, d / fn)
            runs[name] = {"run_dir": str(d), "freq_hz": float(f),
                          "source_field": src.upper(), "meta": m}
    (fwd / "manifest.json").write_text(json.dumps(
        {"mode": "per_frequency_per_source", "source_fields": [s.upper() for s in sources],
         "flist_hz": [float(f) for f in freqs], "n_datasets": len(runs),
         "runs": runs}, indent=2) + "\n")
    return fwd


def _is_offending(val, fwd_root: Path) -> bool:
    if not isinstance(val, Path):
        return False
    try:
        if val.parent.resolve() != fwd_root.resolve():
            return False
    except OSError:
        return False
    return val.name in DATASET_ARTIFACTS


def offending_globals(g: dict, fwd_root: Path):
    """Paths that point at a dataset artifact in the forward ROOT.

    Checks bare `Path` globals AND the attributes of objects held in globals.
    The second half is not optional: the notebooks now bind ONE
    `headless.DatasetPaths` (`DS`) instead of a dozen loose path constants, so
    a bare-`Path`-only walk would report PASS on a notebook whose every dataset
    artifact was bound to the root. A check that cannot see the thing it checks
    is worse than no check, because it reads as evidence.
    """
    bad = []
    for name, val in g.items():
        if name.startswith("__"):
            continue
        if _is_offending(val, fwd_root):
            bad.append(f"{name} = <forward root>/{val.name}")
            continue
        # One level of attributes, for dataclasses / SimpleNamespace holding paths.
        fields = getattr(type(val), "__dataclass_fields__", None)
        attrs = list(fields) if fields else (
            list(vars(val)) if isinstance(val, SimpleNamespace) else [])
        for attr in attrs:
            sub = getattr(val, attr, None)
            if _is_offending(sub, fwd_root):
                bad.append(f"{name}.{attr} = <forward root>/{sub.name}")
    return sorted(bad)


def _find_donor(fwd: Path):
    """A directory holding a real `setup_metadata.json` to copy the fixture from.

    Once the workshop's own workspace IS an acquisition matrix, the forward root
    no longer carries one - it holds `manifest.json` and subdirectories. Without
    this fallback the sweep would quietly SKIP from that point on, and a SKIP
    reads like a pass.
    """
    if (fwd / "setup_metadata.json").exists():
        return fwd
    for sub in sorted(p for p in fwd.glob("*") if p.is_dir()):
        if (sub / "setup_metadata.json").exists():
            return sub
    return None


def run(nb_dir: Path = ROOT):
    from scripts.modules import workshop_config
    donor = _find_donor(ROOT / "workspace" / "2D" / "forward")
    if donor is None:
        # Say this loudly. A quiet "SKIP" next to three PASSes reads as a fourth
        # pass, and this is the one check that catches the matrix path-binding
        # bug - the bug that reached a user. A fresh clone has no workspace, so
        # this is the DEFAULT state, not an edge case.
        print(f"SKIP: no donor dataset under "
              f"{ROOT / 'workspace' / '2D' / 'forward'} to build the fixture from.")
        print("\nRESULT: NOT VERIFIED - the path-binding invariant was NOT tested.")
        print("        Run Step 01 (or scripts/rebuild_matrix_workspace.py) first,")
        print("        then re-run this sweep. Do not read this as a pass.")
        return 0
    print(f"donor dataset: {donor}")

    tmp = Path(tempfile.mkdtemp(prefix="chainsweep_"))
    try:
        fwd_root = build_matrix_workspace(tmp / "workspace", donor)
        real = workshop_config.load_config
        workshop_config.load_config = (
            lambda root=None: dataclasses.replace(real(root),
                                                  workspace_dir=str(tmp / "workspace")))
        from IPython.display import display as _d
        import plotly.graph_objects as _go
        _go.Figure.show = lambda self, *a, **k: None   # never open a browser tab

        print(f"matrix fixture: {fwd_root}")
        print(f"  root contains: {sorted(p.name for p in fwd_root.iterdir())}\n")
        fail = 0
        for nb in NOTEBOOKS:
            code = "\n\n".join("".join(c["source"])
                               for c in json.loads((nb_dir / nb).read_text())["cells"]
                               if c["cell_type"] == "code")
            g = {"__name__": "__main__", "__file__": str(nb_dir / nb), "display": _d}
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(code, nb, "exec"), g)
            except Exception as exc:
                print(f"BUG {nb}: raised on a matrix workspace: "
                      f"{type(exc).__name__}: {exc}")
                fail += 1
                continue
            bad = offending_globals(g, fwd_root)
            if bad:
                print(f"BUG {nb}: dataset artifact bound to the forward ROOT:")
                for b in bad:
                    print(f"      {b}")
                fail += 1
            else:
                print(f"OK  {nb}")
        print("\nRESULT:", "PASS - every notebook resolves onto a real dataset"
              if not fail else f"{fail} notebook(s) break on a matrix workspace")
        return 1 if fail else 0
    finally:
        workshop_config.load_config = real
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(run())
