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


def offending_globals(g: dict, fwd_root: Path):
    """Path globals that point at a dataset artifact in the forward ROOT."""
    bad = []
    for name, val in g.items():
        if not isinstance(val, Path):
            continue
        try:
            if val.parent.resolve() != fwd_root.resolve():
                continue
        except OSError:
            continue
        if val.name in DATASET_ARTIFACTS:
            bad.append(f"{name} = <forward root>/{val.name}")
    return sorted(bad)


def run(nb_dir: Path = ROOT):
    from scripts.modules import workshop_config
    donor = ROOT / "workspace" / "2D" / "forward"
    if not (donor / "setup_metadata.json").exists():
        print(f"SKIP: need a donor dataset at {donor} to build the fixture.")
        return 0

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
