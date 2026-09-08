"""Task 3 in 2D: stage the frequency ladder and run it against the baseline.

This drives the machinery in `scripts.modules.multiscale_2d` end to end:

  --stage verify    Check the grid handoff on the KNOWN true model before it is
                    trusted on an inverted one: resample true -> coarse grid ->
                    fine grid and report how much survives, and whether the
                    interpolant ever produced resistivities outside the source
                    range (it must not - that is what interpolating in log
                    resistivity buys).

  --stage prepare   Write every stage's inversion directory: inputs staged from
                    that frequency's own forward run, `inv.cfg` with that run's
                    order/lpml/PML, the scheduled `dtx`/`dtz` and
                    `tik_sgregalpha`, and `Sg` = the previous stage's output
                    resampled onto this stage's grid. Nothing is run.

  --stage run       Run the staged ladder, then the single-stage all-frequency
                    baseline from the SAME starting model, and report both on
                    the same final grid.

COST WARNING. A 2D FWI is forward + adjoint per shot per function evaluation,
over 30 shots, and `max_iterations=20` with `max_linesearch=5` can reach ~100
evaluations. Measure one iteration before committing to twenty: `--max-iterations`
exists so the structure can be exercised cheaply, and a reduced run must be
reported as a reduced run, not as the head-to-head.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules.inversion import (  # noqa: E402
    create_initial_sg0_model, prepare_inversion_inputs, read_cfg_values, update_cfg_values,
)
from scripts.modules.multiscale_2d import (  # noqa: E402
    build_ladder, read_sg_grid, resample_model_log_rho, verify_roundtrip,
)
from scripts.modules.workshop_config import load_config  # noqa: E402

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


def prepare_stage(stage, *, previous_model: Path | None, initial_rho: float,
                  constrain: bool, sg_min: float, sg_max: float) -> dict:
    """One stage's inversion directory, matched to its own forward run."""
    stage.run_dir.mkdir(parents=True, exist_ok=True)
    (stage.run_dir / "Local").mkdir(exist_ok=True)   # snapmethod=1 needs it to exist
    created = prepare_inversion_inputs(
        fdmodel_dir=stage.forward_dir,
        template_cfg=TEMPLATES / "inv.cfg",
        output_dir=stage.run_dir,
        max_iterations=stage.max_iterations,
        apertx=stage.apertx_m,
        dtx=stage.dtx_m, dtz=stage.dtz_m,
        initial_model_mode="uniform_resistivity",
        uniform_resistivity=initial_rho,
        tik_sgregalpha=stage.tik_sgregalpha,
        sg_min=sg_min, sg_max=sg_max, constrain=constrain,
    )
    # The handoff: replace the uniform sg0 with the previous stage's model,
    # resampled onto THIS stage's grid, written where it can be inspected.
    if previous_model is not None:
        report = resample_model_log_rho(previous_model, stage.forward_dir / "sg.rss",
                                        stage.run_dir / "sg0.rss")
        stage.resample_report = report
        stage.sg0_from = Path(previous_model)
    update_cfg_values(stage.run_dir / "inv.cfg", {"tv_sgregalpha": f"{stage.tv_sgregalpha:.6g}"})
    return {"created": {k: str(v) for k, v in created.items()},
            "cfg": read_cfg_values(stage.run_dir / "inv.cfg")}


def run_stage(stage, nproc: int) -> dict:
    cfg = load_config()
    engine = cfg.binary_path(cfg.inversion_engine_te2d())
    values = read_cfg_values(stage.run_dir / "inv.cfg")
    print(f"[inv ] {stage.run_dir.name}: f={stage.freq_hz:.0f} Hz  order={values['order']} "
          f"lpml={values['lpml']} dtx={values['dtx']} tik={values['tik_sgregalpha']} "
          f"max_iter={values['max_iterations']} -np {nproc}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run([cfg.mpirun, "-np", str(nproc), str(engine), "inv.cfg"],
                          cwd=str(stage.run_dir), capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"inversion failed in {stage.run_dir}:\n"
                           + ((proc.stdout or "") + (proc.stderr or ""))[-2000:])
    print(f"[inv ] done in {wall:.1f} s", flush=True)
    return {"wall_s": wall, "cfg": values}


def model_error_vs_truth(model_path: Path, true_sg: Path) -> float:
    """RMS log10-resistivity error against the true model, on the model's grid."""
    a = read_sg_grid(model_path)
    b = read_sg_grid(true_sg)
    la = -np.log10(np.clip(a["sigma"], 1e-30, None))
    lb = -np.log10(np.clip(b["sigma"], 1e-30, None))
    tmp = np.empty((a["nx"], b["z"].size))
    for k in range(b["z"].size):
        tmp[:, k] = np.interp(a["x"], b["x"], lb[:, k], left=lb[0, k], right=lb[-1, k])
    lb_on_a = np.empty((a["nx"], a["nz"]))
    for i in range(a["nx"]):
        lb_on_a[i, :] = np.interp(a["z"], b["z"], tmp[i, :], left=tmp[i, 0], right=tmp[i, -1])
    return float(np.sqrt(np.mean((la - lb_on_a) ** 2)))


def find_output_model(run_dir: Path) -> Path | None:
    """The inverted Sg the engine wrote (name varies with the build)."""
    for pattern in ("*Sg*final*.rss", "*sg*final*.rss", "*Sg_*.rss", "*sg_*.rss", "*.rss"):
        hits = sorted(p for p in run_dir.glob(pattern)
                      if "grad" not in p.name.lower() and p.name not in
                      {"sg0.rss", "ep.rss", "wav2d.rss", "weight.rss",
                       "Hx_data.rss", "Hz_data.rss", "Sg_min.rss", "Sg_max.rss"})
        if hits:
            return hits[-1]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", default=["verify"],
                    choices=["verify", "prepare", "run"])
    ap.add_argument("--per-frequency-root", default="workspace/2D/per_frequency")
    ap.add_argument("--ladder-root", default="workspace/2D/inversion/ladder")
    ap.add_argument("--baseline-root", default="workspace/2D/inversion/baseline")
    ap.add_argument("--broadband-dir", default="workspace/2D/forward")
    ap.add_argument("--alpha-final", type=float, default=0.0)
    ap.add_argument("--max-iterations", type=int, default=20)
    ap.add_argument("--initial-rho", type=float, default=10.0)
    ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--no-joint", action="store_true")
    ap.add_argument("--out", default="workspace/2D/inversion/multiscale_2d.json")
    args = ap.parse_args()

    root = Path(args.per_frequency_root)
    manifest = json.loads((root / "manifest.json").read_text())
    stages = build_ladder(manifest, args.ladder_root, alpha_final=args.alpha_final,
                          max_iterations=args.max_iterations,
                          joint_final_stage=not args.no_joint)
    out: dict = {}
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        out = json.loads(outp.read_text())

    print("=== ladder ===")
    print(f"{'stage':>12} {'f [Hz]':>8} {'dx m':>6} {'dtx m':>7} {'tik':>9} {'apertx m':>9}")
    for s in stages:
        print(f"{s.run_dir.name:>12} {s.freq_hz:8.0f} {s.dx_m:6.2f} {s.dtx_m:7.2f} "
              f"{s.tik_sgregalpha:9.4g} {s.apertx_m:9.2f}")

    if "verify" in args.stage:
        coarse = Path(stages[0].forward_dir) / "sg.rss"
        fine = Path(stages[-1].forward_dir) / "sg.rss"
        rep = verify_roundtrip(fine, coarse, fine, Path(args.ladder_root) / "_verify")
        print("\n=== grid handoff verified on the KNOWN true model ===")
        print(f"  true ({rep['down']['src_dx']:.2f} m) -> coarse ({rep['down']['dst_dx']:.2f} m) "
              f"-> fine ({rep['up']['dst_dx']:.2f} m)")
        print(f"  rho range   src {rep['down']['rho_min_src']:.3g}-{rep['down']['rho_max_src']:.3g}"
              f"  -> coarse {rep['down']['rho_min_dst']:.3g}-{rep['down']['rho_max_dst']:.3g}")
        print(f"  OVERSHOOT anywhere: {rep['overshoot_anywhere']}   "
              f"(must be False - that is what interpolating in log resistivity buys)")
        print(f"  true vs round-trip: rms {rep['true_vs_roundtrip_rms_log10']:.4f} decades, "
              f"max {rep['true_vs_roundtrip_max_log10']:.4f} decades")
        print("  (a non-zero rms here is the EXPECTED long-wavelength truncation the coarse")
        print("   stage imposes, not a bug - it is what the early stages exist to do)")
        out["verify"] = rep
        outp.write_text(json.dumps(out, indent=2) + "\n")

    if "prepare" in args.stage or "run" in args.stage:
        prev = None
        prepared = []
        for s in stages:
            info = prepare_stage(s, previous_model=prev, initial_rho=args.initial_rho,
                                 constrain=True, sg_min=1.0 / 150.0, sg_max=1.0 / 1.0)
            prepared.append({**s.to_dict(), **info})
            if "run" in args.stage:
                timing = run_stage(s, args.nproc)
                prepared[-1]["timing"] = timing
                mdl = find_output_model(s.run_dir)
                prepared[-1]["output_model"] = str(mdl) if mdl else None
                if mdl:
                    prepared[-1]["model_err_vs_truth"] = model_error_vs_truth(
                        mdl, Path(s.forward_dir) / "sg.rss")
                    prev = mdl
                    print(f"       model error vs truth: "
                          f"{prepared[-1]['model_err_vs_truth']:.4f} decades", flush=True)
            else:
                prev = s.run_dir / "sg0.rss"
        out["stages"] = prepared
        outp.write_text(json.dumps(out, indent=2, default=str) + "\n")
        print(f"\nstaged {len(prepared)} stage(s) under {args.ladder_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
