#!/usr/bin/env python3
"""Acceptance test for JOINT multi-source 2D FWI, through the workshop's chain.

The workshop acquires a 4-component magnetic tensor as (2 sources) x (2
receivers) - `Cxx = Hx<-Kx`, `Cxz = Hz<-Kx`, `Czx = Hx<-Kz`, `Czz = Hz<-Kz`.
Each FDTD run records BOTH receiver components but fires ONE source, so the
dataset is two forward runs. `mpiEminvTE2d` used to take one `source_type` per
run, so the workshop CASCADED over sources; it now takes a comma-separated list
and inverts them jointly, and the whole change reduces to one number:

    TEST 1 (acceptance)   g(joint) == g(Kx) + g(Kz)   to float round-off.

`Image2D::stackImage` accumulates in float and the joint run sums 2*ngathers
per-item images in a different order than two runs summing ngathers each, so
this is round-off equality, not bit equality.

This is the workshop's own version of
`doc/examples/validate_joint_source_gradient/run_joint_gradient.py` in
rockem-suite, and the model, acquisition and tolerances are deliberately the
same. The difference is what is being tested: the three inversion directories
are staged by `inversion.prepare_inversion_inputs` from
`scripts/templates/inv.cfg`, on forward directories shaped exactly like the ones
`headless.build_forward_matrix` writes - so this covers the workshop's staging
chain (record-file naming, the `Recordfile_<SRC>_<REC>` keys, the source_type
list, the mod.cfg assertions), not only the engine.

A TINY dedicated model, not the shipped 2 kHz dataset: the equality being tested
is shot-count independent, and the production grids cost ~30-45 minutes for the
same answer. This runs in a few minutes.

Run:
    python3 scripts/experiments/joint_source_gradient.py
    python3 scripts/experiments/joint_source_gradient.py --keep   # keep the dirs
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.modules import rockem_bridge  # noqa: E402  (puts rockem.* on sys.path)
from scripts.modules.inversion import (  # noqa: E402
    prepare_inversion_inputs, read_cfg_values,
)
from scripts.modules.workshop_config import load_config  # noqa: E402
from third_party.rockseis.io.rsfile import rsfile  # noqa: E402

LayerSpec = rockem_bridge.model.LayerSpec
build_layered_1d_grid = rockem_bridge.model.build_layered_1d_grid
write_model_rss = rockem_bridge.model.write_model_rss
write_survey = rockem_bridge.survey.write_survey
write_te2d_config = rockem_bridge.config.write_te2d_config
Wavelet = rockem_bridge.wavelet.Wavelet

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"

# --- Model / acquisition ----------------------------------------------------
# Mirrors the upstream reference exactly. A 1D LAYERED model on purpose: it is
# the regime the workshop inverts. order=8 at lpml=14 is a known-good pair here;
# order=6 at lpml=13 makes the PML go unstable in this diffusive regime and the
# gradient comes back all-NaN - on the pre-change binary too, i.e. a parameter
# trap, not an engine bug.
DOMAIN_M = [300.0, 300.0]
EPS_R = 7.0
LAYERS = (
    LayerSpec(resistivity_ohm_m=100.0, permittivity=EPS_R, thickness_m=60.0),
    LayerSpec(resistivity_ohm_m=60.0, permittivity=EPS_R, thickness_m=40.0),
    LayerSpec(resistivity_ohm_m=140.0, permittivity=EPS_R, thickness_m=None),
)
START_RHO_OHM_M = 110.0      # homogeneous start -> a nonzero, meaningful gradient
F0_HZ = 4000.0
LPML = 14
ORDER = 8
OFFSETS_M = np.array([40.0, 70.0, 100.0])
SHOT_X_OFFSETS = np.array([-30.0, 30.0])   # >1 shot, so the (shot, source) decode is exercised
ORDER_CFL_SCALE = {2: 1.0, 4: 0.65, 8: 0.42}

SOURCES = ("HX", "HZ")
RECEIVERS = ("HX", "HZ")
# The filenames a workshop forward run writes, so `prepare_data_from_fdmodel`
# consumes these directories unchanged.
FORWARD_RECORDS = {"HX": "Data/Hxshot.rss", "HZ": "Data/Hzshot.rss"}

GRAD_TOL = 1e-5      # relative L2, joint vs sum of singles (float stacking)
COST_TOL = 1.25      # joint / (single_a + single_b)
NPROC = 2


def build_params() -> dict:
    utils = rockem_bridge.utils
    sigma_min = min(1.0 / l.resistivity_ohm_m for l in LAYERS)
    sigma_max = max(1.0 / l.resistivity_ohm_m for l in LAYERS)
    eps_min = min(l.permittivity for l in LAYERS)
    eps_max = max(l.permittivity for l in LAYERS)
    dx = utils.suggest_grid_size(utils.GridRecInputs(
        f_min_hz=F0_HZ, f_max_hz=F0_HZ,
        sigma_min_s_per_m=sigma_min, sigma_max_s_per_m=sigma_max,
        eps_r_min=eps_min, eps_r_max=eps_max,
        points_per_skin=12, cells_per_wavelength=8,
    )).delta_recommended_m
    dt = utils.suggest_time_steps(utils.TimeRecInputs(
        grid=utils.GridSpec(dx=dx, dz=dx), eps_r_min=eps_min, sigma_max_s_per_m=sigma_max,
        explicit_cfl_safety=0.95, adi_k_cfl=10.0, use_grid_diff_cap=True,
    )).dt_explicit_cfl_s * ORDER_CFL_SCALE[ORDER]
    vmax = 3e8 / np.sqrt(eps_min)
    pml_out = utils.suggest_pml_parameters(utils.PmlRecInputs(
        lpml=LPML, dx=dx, dz=dx, f0_hz=F0_HZ, vmax_m_per_s=vmax,
        kappa_max=11.0, alpha_fraction_of_omega0=1.0,
    ))
    # apertx > 0 is a SOURCE-CENTRED TOTAL WIDTH taken from each gather's own
    # coordinates. Both source types sit at the same coordinates, so both see
    # the same local model - which is exactly what the joint gradient needs.
    apertx = 2.0 * OFFSETS_M.max() + 80.0
    # Long enough that the PML interaction has died before the record ends.
    nt = int(round((DOMAIN_M[0] / 2 + 2 * LPML * dx) / vmax / dt)) * 12
    return {"dx": dx, "dt": dt, "pml": pml_out, "apertx": apertx, "nt": nt}


def run_binary(binary: str, cfg_name: str, cwd: Path, label: str) -> float:
    cfg = load_config()
    t0 = time.perf_counter()
    proc = subprocess.run(
        [cfg.mpirun, "-np", str(NPROC), str(cfg.binary_path(binary)), cfg_name],
        cwd=str(cwd), capture_output=True, text=True,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(((proc.stdout or "") + (proc.stderr or ""))[-4000:])
        raise RuntimeError(f"{binary} failed for {label} in {cwd}")
    print(f"  {label:<14} {wall:7.1f} s", flush=True)
    return wall


def build_forward_dirs(root: Path, p: dict) -> dict:
    """One forward directory per source, shaped like a workshop forward run.

    `sg.rss`, `ep.rss`, `wav2d.rss`, `mod.cfg` and `Data/{Hx,Hz}shot.rss` - the
    exact layout `headless.build_forward_matrix` produces, so
    `prepare_inversion_inputs` consumes them with no special-casing.

    Every source shares one survey and one wavelet, which is what makes the four
    record files agree on trace count, order, coordinates and time axis - the
    engine's hard requirement for a joint run, which it checks at startup
    (src/mpiEminvTE2d.cpp:553-585).
    """
    dx, dt = p["dx"], p["dt"]
    tx_z = DOMAIN_M[1] / 2
    cx = DOMAIN_M[0] / 2

    grid = build_layered_1d_grid(
        layers=LAYERS, domain_size_m=DOMAIN_M, dx=dx, tx_depth_m=tx_z, dim=2)
    src_x, src_z, rx_x, rx_z = [], [], [], []
    for sx_off in SHOT_X_OFFSETS:
        for off in OFFSETS_M:
            src_x.append(cx + sx_off); src_z.append(tx_z)
            rx_x.append(cx + sx_off + off); rx_z.append(tx_z)
    wav = Wavelet()
    wav.Ramp_cw(F0_HZ, 4, p["nt"], dt)

    dirs = {}
    for src in SOURCES:
        d = root / src.lower()
        (d / "Data").mkdir(parents=True, exist_ok=True)
        write_model_rss(grid, str(d / "sg.rss"), str(d / "ep.rss"))
        write_survey(str(d / "Survey.rss"), src_x=src_x, src_z=src_z,
                     rx_x=rx_x, rx_z=rx_z, dim=2)
        wav.write(str(d / "wav2d.rss"), dim=2, src_x=cx, src_z=tx_z)
        write_te2d_config(
            str(d / "mod.cfg"),
            sg_file="sg.rss", ep_file="ep.rss", wavelet_file="wav2d.rss",
            survey_file="Survey.rss", source_field=src,
            order=ORDER, lpml=LPML, adi=False, usepml=True,
            snapinc=100000, dtrec=dt, apertx=p["apertx"],
            pml_kmax=p["pml"].pml_kmax, pml_smax=p["pml"].pml_smax,
            pml_amax=p["pml"].pml_amax,
            records=RECEIVERS, recordfiles=dict(FORWARD_RECORDS),
        )
        run_binary(load_config().forward_engine_te2d(), "mod.cfg", d, f"forward {src}")
        dirs[src] = d
    return dirs


def stage(run_dir: Path, forward_dirs: dict, p: dict) -> Path:
    """One inversion directory, staged the way the workshop stages them."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "Local").mkdir(exist_ok=True)     # snapmethod=1 needs it to exist
    (run_dir / "Results").mkdir(exist_ok=True)
    prepare_inversion_inputs(
        fdmodel_dir=forward_dirs,
        template_cfg=TEMPLATES / "inv.cfg",
        output_dir=run_dir,
        # max_iterations = 0 ON PURPOSE. `runGrad` overwrites sg_grad.rss and
        # misfit.rss on EVERY function evaluation, and `Opt::opt_lbfgs` does the
        # initial evaluate(current) BEFORE `for(i=0; i<max_iterations; i++)`, so
        # 0 leaves exactly ONE evaluation, at the starting model. With 1 each run
        # also does line-search evaluations and sg_grad.rss ends up being the
        # gradient at whatever trial model that run's own line search reached;
        # the three runs take different steps, so the comparison would be between
        # gradients at DIFFERENT models - a few-percent discrepancy that looks
        # exactly like a bug in the joint code and is not one.
        max_iterations=0,
        apertx=p["apertx"], dtx=6.0 * p["dx"], dtz=6.0 * p["dx"],
        initial_model_mode="uniform_resistivity", uniform_resistivity=START_RHO_OHM_M,
        tik_sgregalpha=0.0,
        sg_min=1.0 / 500.0, sg_max=1.0 / 10.0, constrain=True,
    )
    values = read_cfg_values(run_dir / "inv.cfg")
    # Regularisation MUST be off. g = g_data + g_reg, and a model-only g_reg is
    # counted twice on the right-hand side of g_HX + g_HZ but once in g_joint,
    # so any non-zero penalty breaks the identity being tested for a reason that
    # has nothing to do with joint sources.
    for key in ("tik_sgregalpha", "tv_sgregalpha", "tik_epregalpha", "tv_epregalpha"):
        if float(values[key]) != 0.0:
            raise RuntimeError(f"{key} = {values[key]} in {run_dir}; the gradient "
                               "identity only holds with no regularisation term.")
    return run_dir / "inv.cfg"


def assert_single_evaluation(run_dir: Path) -> None:
    """`max_iterations = 0` must leave exactly ONE evaluation, at the start model."""
    log = run_dir / "progress.log"
    if not log.exists():
        raise RuntimeError(f"No progress.log in {run_dir}")
    text = log.read_text()
    lines = [l for l in text.splitlines() if l.startswith("Linesearch")]
    if len(lines) != 1:
        raise RuntimeError(
            f"{run_dir.name}: expected ONE Linesearch line (one evaluation at the "
            f"starting model), got {len(lines)}:\n  " + "\n  ".join(lines))
    # Columns are ITERATION, STEP LENGTH, MISFIT, GNORM, MNORM, TIME. A zero step
    # length and a zero model norm together say the evaluation happened at the
    # starting model, before the optimiser moved anywhere.
    cols = lines[0].split()
    step, mnorm = float(cols[1]), float(cols[4])
    if step != 0.0 or mnorm != 0.0:
        raise RuntimeError(f"{run_dir.name}: the single evaluation is not at the "
                           f"starting model (STEP LENGTH={step}, MNORM={mnorm})")
    # With max_iterations = 0 the optimiser reports this, and it is expected.
    if "Optimization method not started" not in text:
        raise RuntimeError(f"{run_dir.name}: expected 'Optimization method not started' "
                           "in progress.log with max_iterations = 0")


def load(path: Path) -> np.ndarray:
    f = rsfile(); f.read(str(path))
    return np.asarray(f.data, dtype=np.float64)


def rel_l2(ref: np.ndarray, test: np.ndarray) -> float:
    den = np.linalg.norm(ref)
    return float(np.linalg.norm(test - ref) / den) if den > 0 else float(np.linalg.norm(test - ref))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="working directory (default: workspace/2D/joint_gradient_check)")
    ap.add_argument("--keep", action="store_true", help="do not delete the run directories first")
    args = ap.parse_args()

    root = Path(args.root) if args.root else \
        Path(load_config().workspace) / "2D" / "joint_gradient_check"
    if not args.keep and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    p = build_params()
    print(f"tiny model: dx={p['dx']:.2f} m  dt={p['dt']:.3e} s  nt={p['nt']}  "
          f"apertx={p['apertx']:.1f} m  order={ORDER} lpml={LPML}")
    print("\nForward-modelling the true layered model, one run per source type ...")
    forward_dirs = build_forward_dirs(root / "forward", p)

    print("\nStaging three inversions through prepare_inversion_inputs ...")
    runs = {
        "HX": stage(root / "run_hx", {"HX": forward_dirs["HX"]}, p),
        "HZ": stage(root / "run_hz", {"HZ": forward_dirs["HZ"]}, p),
        "joint": stage(root / "run_joint", forward_dirs, p),
    }
    for label, cfg in runs.items():
        v = read_cfg_values(cfg)
        keys = sorted(k for k in v if k.startswith("Recordfile") and v[k])
        print(f"  {label:<6} source_type={v['source_type']:<5} {' '.join(keys)}")

    print("\nRunning one gradient evaluation per configuration ...")
    walls = {label: run_binary(load_config().inversion_engine_te2d(), "inv.cfg",
                               cfg.parent, label)
             for label, cfg in runs.items()}
    for cfg in runs.values():
        assert_single_evaluation(cfg.parent)
    print("  every run did exactly ONE evaluation, at the starting model "
          "(STEP LENGTH = 0, MNORM = 0)")

    ok = True

    # --- TEST 1: gradient equivalence (the acceptance test) ------------------
    g = {k: load(runs[k].parent / "sg_grad.rss") for k in ("HX", "HZ", "joint")}
    if not (g["HX"].shape == g["HZ"].shape == g["joint"].shape):
        raise RuntimeError(f"gradient shape mismatch: "
                           f"{g['HX'].shape} {g['HZ'].shape} {g['joint'].shape}")
    g_sum = g["HX"] + g["HZ"]
    err = rel_l2(g_sum, g["joint"])
    maxrel = float(np.abs(g["joint"] - g_sum).max() / max(np.abs(g_sum).max(), 1e-300))
    print("\n=== TEST 1: joint gradient == sum of single-source gradients ===")
    print(f"  ||g_joint - (g_HX + g_HZ)|| / ||g_HX + g_HZ||  = {err:.3e}   (tol {GRAD_TOL:.0e})")
    print(f"  max|diff| / max|g_HX + g_HZ|                   = {maxrel:.3e}")
    print(f"  ||g_HX||={np.linalg.norm(g['HX']):.6e}  ||g_HZ||={np.linalg.norm(g['HZ']):.6e}  "
          f"||g_joint||={np.linalg.norm(g['joint']):.6e}")
    if not np.isfinite(err) or err >= GRAD_TOL:
        print("  FAIL"); ok = False
    elif np.linalg.norm(g["joint"]) == 0.0:
        print("  FAIL: gradient is identically zero - the test proves nothing"); ok = False
    else:
        print("  PASS")

    # `misfit.rss` is ngathers*nsources entries, SOURCE-MAJOR, and the total is
    # the plain sum over all of them.
    m = {k: load(runs[k].parent / "misfit.rss") for k in ("HX", "HZ", "joint")}
    m_err = abs(m["joint"].sum() - (m["HX"].sum() + m["HZ"].sum())) / \
        max(abs(m["HX"].sum() + m["HZ"].sum()), 1e-300)
    print(f"  misfit entries: HX={m['HX'].size} HZ={m['HZ'].size} joint={m['joint'].size} "
          f"(joint must be the two singles' counts summed)")
    print(f"  misfit total: joint={m['joint'].sum():.8e}  HX+HZ="
          f"{m['HX'].sum() + m['HZ'].sum():.8e}  rel diff={m_err:.3e}")
    if m["joint"].size != m["HX"].size + m["HZ"].size:
        print("  FAIL (misfit entry count)"); ok = False
    if m_err >= GRAD_TOL:
        print("  FAIL (misfit total)"); ok = False

    # --- TEST 2: single-source staging is unchanged --------------------------
    print("\n=== TEST 2: single-source runs are untouched ===")
    v = read_cfg_values(runs["HX"])
    pair_keys = [k for k in v if k.startswith("Recordfile_HX_") or k.startswith("Recordfile_HZ_")]
    staged = sorted(x.name for x in runs["HX"].parent.glob("*_data.rss"))
    outputs = sorted(x.name for x in runs["HX"].parent.glob("data_*.rss"))
    print(f"  source_type={v['source_type']!r}  record keys="
          f"{sorted(k for k in v if k.startswith('Recordfile') and v[k])}")
    print(f"  staged data: {staged}")
    print(f"  engine wrote: {outputs}")
    single_ok = (v["source_type"] == "3" and not pair_keys
                 and staged == ["Hx_data.rss", "Hz_data.rss"]
                 and outputs == ["data_mod_HX.rss", "data_mod_HZ.rss",
                                 "data_res_HX.rss", "data_res_HZ.rss"])
    print("  PASS" if single_ok else "  FAIL"); ok = ok and single_ok

    # --- TEST 3: joint outputs carry the source tag --------------------------
    print("\n=== TEST 3: joint outputs are tagged by source ===")
    joint_out = sorted(x.name for x in runs["joint"].parent.glob("data_*.rss"))
    print(f"  engine wrote: {joint_out}")
    expected = sorted(f"data_{kind}_{s}_{r}.rss"
                      for kind in ("mod", "res") for s in SOURCES for r in RECEIVERS)
    tag_ok = joint_out == expected
    print("  PASS" if tag_ok else f"  FAIL (expected {expected})"); ok = ok and tag_ok

    # --- TEST 4: cost --------------------------------------------------------
    ratio = walls["joint"] / max(walls["HX"] + walls["HZ"], 1e-9)
    print("\n=== TEST 4: cost ===")
    print(f"  t(joint) / (t(HX) + t(HZ))    = {ratio:.2f}   (tol {COST_TOL:.2f})")
    print(f"  t(joint) / mean single-source = "
          f"{walls['joint'] / max(0.5 * (walls['HX'] + walls['HZ']), 1e-9):.2f}   "
          "(~2x for two sources, not 4x)")
    print("  One forward/adjoint pair per (shot, source); both receiver components "
          "come from the same propagation.")
    print("  PASS" if ratio < COST_TOL else "  FAIL"); ok = ok and ratio < COST_TOL

    print("\n" + ("ALL TESTS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
