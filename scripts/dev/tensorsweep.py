#!/usr/bin/env python3
"""Does the 4-COMPONENT chain carry the right data end to end?

The bug class this exists to catch, and which no other sweep can see:

    A joint multi-source inversion puts all four tensor components - Cxx, Cxz,
    Czx, Czz - in ONE run directory, as four `Recordfile_<SRC>_<REC>` inputs and
    four `data_mod_<SRC>_<REC>` outputs. The view selector in Step 04 has a
    SOURCE axis, so picking "Kz -> Hx (Czx)" must read the Kz source's files.
    Nothing about a reader that silently returns the Kx pair is a NameError, a
    failed exec, a bad path or a false sentence, so `handlersweep`, `bugsweep`,
    `chainsweep` and `docsweep` all PASS while the panel plots Cxx under a Czx
    label. That shipped once; this is the check that would have caught it.

What it asserts, against a REAL joint run directory:

1. Every (source, receiver) view resolves to the observed file that its own
   source wrote - four distinct files over the four views, never one pair twice.
2. The same for the modelled/residual data the engine wrote.
3. The component label matches the (source, receiver) pair.
4. TX/RX marker positions resolve (a hardcoded `Hx_data.rss` returns none on a
   joint run, and the model plots lose their geometry silently).
5. Staged inputs and engine outputs are self-consistent: one record file per
   (source, receiver), `source_type` listing exactly those sources.

Run:  python3 scripts/dev/tensorsweep.py [run_dir]
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import plotly.io as pio
pio.renderers.default = "json"
import plotly.graph_objects as _go
_go.Figure.show = lambda self, *a, **k: None

from scripts.modules.inversion import (  # noqa: E402
    available_synthetic_pairs, find_observed_records, read_cfg_values,
    source_fields_from_cfg, staged_record_name,
)
from scripts.modules.multiscale_2d import iter_stages, list_run_dirs  # noqa: E402
from scripts.modules.workshop_config import load_config  # noqa: E402

RECEIVERS = ("HX", "HZ")
LABEL = {("HX", "Hx"): "Cxx", ("HX", "Hz"): "Cxz",
         ("HZ", "Hx"): "Czx", ("HZ", "Hz"): "Czz"}
failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"OK    {msg}")


def find_joint_run() -> Path | None:
    """The newest joint SCALE directory the ENGINE has written output into.

    A ladder's engine cwd is `Run{N}/<freq>/`, not the Run root. A stage
    staged but never executed has the four record files and no `data_mod_*`;
    executed stages are preferred. If none exists the newest staged joint
    scale is returned, and the modelled-data checks report SKIP.
    """
    runs_root = Path(load_config().inv_2d_runs_dir)
    if not runs_root.exists():
        return None
    cands: list[Path] = []
    for _, ladder in list_run_dirs(runs_root):
        for st in iter_stages(ladder):
            d = Path(st["path"])
            if (d / "inv.cfg").exists() and len(source_fields_from_cfg(d)) > 1:
                cands.append(d)
    if not cands:
        return None
    executed = [d for d in cands if any(d.glob("data_mod_*.rss*"))]
    pool = executed or cands
    return sorted(pool, key=lambda d: d.stat().st_mtime)[-1]


def check_staging(run_dir: Path, sources: list[str]) -> None:
    """One record file per (source, receiver), and source_type lists them all."""
    values = read_cfg_values(run_dir / "inv.cfg")
    listed = values.get("source_type", "")
    expect = ",".join({"EY": "1", "HX": "3", "HZ": "5"}[s] for s in sources)
    if listed != expect:
        fail(f"source_type is {listed!r}, expected {expect!r}")
    else:
        ok(f"source_type = {listed!r} for sources {sources}")

    seen = {}
    for src in sources:
        for rec in RECEIVERS:
            key = f"Recordfile_{src}_{rec}"
            name = values.get(key, "")
            if not name:
                fail(f"{key} is empty in inv.cfg")
                continue
            if not (run_dir / name).exists():
                fail(f"{key} = {name!r} but that file is not in {run_dir.name}/")
                continue
            if name in seen:
                fail(f"{key} and {seen[name]} both point at {name!r}")
            seen[name] = key
            if name != staged_record_name(src, rec, len(sources) > 1):
                fail(f"{key} = {name!r}, not the expected staged name")
    if len(seen) == len(sources) * len(RECEIVERS):
        ok(f"{len(seen)} distinct record files staged, one per (source, receiver)")


def check_readers(run_dir: Path, sources: list[str]) -> None:
    """Each view's source must select its OWN observed and modelled files."""
    obs_seen, syn_seen = {}, {}
    for src in sources:
        obs = find_observed_records(run_dir, source_field=src)
        if not obs:
            fail(f"no observed records resolve for source {src}")
            continue
        names = tuple(p.name for p in (obs["HX"], obs["HZ"]))
        if names in obs_seen:
            fail(f"source {src} reads the SAME observed pair as {obs_seen[names]}: {names}")
        obs_seen[names] = src
        for rec in RECEIVERS:
            got = obs[rec].name
            want = staged_record_name(src, rec, len(sources) > 1)
            if got != want:
                fail(f"view {LABEL[(src, rec.title())]} reads {got!r}, expected {want!r}")

        pairs = available_synthetic_pairs(run_dir, source_field=src)
        if not pairs:
            if not any(run_dir.glob("data_mod_*.rss*")):
                print(f"SKIP  {run_dir.name} has no engine output yet - "
                      f"modelled-data checks need a run that has been executed")
                return
            fail(f"no modelled data resolves for source {src}")
            continue
        _, hx, hz = pairs[-1]
        syn = (hx.name, hz.name)
        if syn in syn_seen:
            fail(f"source {src} reads the SAME modelled pair as {syn_seen[syn]}: {syn}")
        syn_seen[syn] = src
        for name, rec in ((hx.name, "HX"), (hz.name, "HZ")):
            if len(sources) > 1 and f"_{src}_{rec}" not in name:
                fail(f"modelled file {name!r} is not source {src}'s {rec}")
    if len(obs_seen) == len(sources):
        ok(f"observed data: {len(obs_seen)} distinct pairs for {len(sources)} sources")
        for names, src in obs_seen.items():
            print(f"        {src}: {names[0]}, {names[1]}")
    if len(syn_seen) == len(sources):
        ok(f"modelled data: {len(syn_seen)} distinct pairs for {len(sources)} sources")
        for names, src in syn_seen.items():
            print(f"        {src}: {names[0]}, {names[1]}")


def check_notebook_04(run_dir: Path, sources: list[str]) -> None:
    """Drive Step 04's own readers through every view of the selector."""
    nb = json.loads((ROOT / "04_2d_inversion_results.ipynb").read_text())
    g = {"__name__": "__main__", "display": lambda *a, **k: None}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        import IPython.display as D
        D.display = lambda *a, **k: None
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                exec(compile("".join(c["source"]), "<nb04>", "exec"), g)

    ladder = run_dir.parent
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        g["refresh_run_options"]()
        g["run_selector"].value = str(ladder)
        g["refresh_scale_options"]()
        g["scale_selector"].value = str(run_dir)
        g["refresh_view_combos"]()

    tx_x, _, rx_x, _ = g["_extract_positions"](run_dir)
    if tx_x.size == 0 or rx_x.size == 0:
        fail("Step 04 resolves NO TX/RX positions in the joint run "
             "(model plots lose their geometry markers)")
    else:
        ok(f"Step 04 TX/RX markers: {tx_x.size} TX, {rx_x.size} RX")

    combos = g["state"]["view_combos"]
    per_view = {}
    for i, k in enumerate(combos):
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            g["view_combo"].value = i
            hx, hz = g["get_real_data_paths"](run_dir, source_field=g["view_source"]())
            label = g["view_component_label"]()
        rec = k["receiver"]
        want_label = LABEL[(k["source_field"], rec)]
        if label != want_label:
            fail(f"view {k['source_field']}->{rec} is labelled {label!r}, expected {want_label!r}")
        chosen = hx if rec == "Hx" else hz
        want = staged_record_name(k["source_field"], "HX" if rec == "Hx" else "HZ",
                                  len(sources) > 1)
        per_view.setdefault(want_label, chosen.name)
        if chosen.name != want:
            fail(f"Step 04 view {want_label} reads {chosen.name!r}, expected {want!r}")
    if len(set(per_view.values())) == len(per_view):
        ok(f"Step 04: all {len(per_view)} tensor views read distinct files: "
           + ", ".join(f"{k}={v}" for k, v in sorted(per_view.items())))


def check_run_model_link() -> None:
    """Does each Step select the true model / grid of the SCALE, not dataset 0?

    Every frequency is modelled on its own grid, so `sg.rss` is a different
    array per dataset. Step 03 plots the true model beside the inversion and
    takes its colour limits from it; Step 04 rebinds `DS` from the selected
    scale. Both must follow the selected stage.
    """
    ladders = [p for _, p in list_run_dirs(load_config().inv_2d_runs_dir)]
    stages = []
    for ladder in ladders:
        for st in iter_stages(ladder):
            if (st["path"] / "dataset.txt").exists():
                stages.append((ladder, st))
    if not stages:
        print("SKIP  no ladder stages with dataset.txt to check the run->model link against")
        return

    buf = io.StringIO()
    g3, g4 = {}, {}
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        import IPython.display as D
        D.display = lambda *a, **k: None
        for nbname, g in (("03_2d_inversion.ipynb", g3),
                          ("04_2d_inversion_results.ipynb", g4)):
            g.update({"__name__": "__main__", "display": lambda *a, **k: None})
            nb = json.loads((ROOT / nbname).read_text())
            for c in nb["cells"]:
                if c["cell_type"] == "code":
                    exec(compile("".join(c["source"]), f"<{nbname}>", "exec"), g)

    bad = 0
    for ladder, st in stages:
        tag = (st["path"] / "dataset.txt").read_text().strip().split("Hz")[0]
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            tm3 = g3["run_true_model"](st["path"])
            g4["refresh_run_options"]()
            g4["run_selector"].value = str(ladder)
            g4["refresh_scale_options"]()
            g4["scale_selector"].value = str(st["path"])
            g4["on_scale_selected"]({"name": "value", "new": str(st["path"])})
            tm4 = g4["DS"].sg
        for step, tm in (("03", tm3), ("04", tm4)):
            if tag not in tm.parent.name:
                fail(f"Step {step}: scale {st['path'].name} ({tag}Hz) reads true model "
                     f"{tm.parent.name}/sg.rss")
                bad += 1
    if not bad:
        ok(f"run -> true model: all {len(stages)} scales read their own dataset in Steps 03 and 04")


def main() -> int:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else find_joint_run()
    if run_dir is None:
        print("SKIP: no joint (multi-source) run directory found under "
              f"{load_config().inv_2d_runs_dir}.")
        print("      Stage and run one from Step 03, or pass a run directory.")
        return 0
    sources = source_fields_from_cfg(run_dir)
    print(f"joint run: {run_dir}")
    print(f"sources:   {sources}\n")

    check_staging(run_dir, sources)
    check_readers(run_dir, sources)
    check_notebook_04(run_dir, sources)
    check_run_model_link()

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S)")
        return 1
    print("RESULT: PASS - every tensor component reads its own data end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
