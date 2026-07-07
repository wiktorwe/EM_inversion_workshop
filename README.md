# EM Inversion Workshop

This repository contains an interactive workshop for electromagnetic (EM) forward modelling, inversion, and result analysis.

## 1) Prerequisites

Install these before starting:

- Python 3.10+
- `mpirun` (OpenMPI or compatible MPI runtime)
- A validated rockem-suite checkout, built (`make mpi`), with the **explicit**
  TE2D engine binaries at:
  `$ROCKEM_SUITE_ROOT/bin/mpiEmmodTE2d`
  `$ROCKEM_SUITE_ROOT/bin/mpiEminvTE2d`
  (`ROCKEM_SUITE_ROOT` defaults to `~/software/new_rockem/rockem-suite` -
  see `scripts/modules/rockem_bridge.py`. The ADI TE2D engine,
  `mpiEmmodADITE2d`/`mpiEminvADITE2d`, is NOT used by this workshop - it
  fails rockem-suite's own layered-model Green's-function validation.)

**Optional — GPU (2D TE2D only):** On an NVIDIA machine, build GPU binaries in rockem-suite:

```bash
make -f Makefile.gpu MPI_CXX=mpic++ RS_CUDA_ARCH=<arch> CUDA_HOME=$CUDA_HOME \
  bin/mpiEmmodTE2dGpu bin/mpiEminvTE2dGpu
```

Enable them in **Step 00** (`use_gpu_forward_2d` / `use_gpu_inversion_2d` in
`workshop_config.json`). Validation checks `nvidia-smi` and GPU binaries.
After enabling GPU forward, re-run **Step 01 → Finalize setup** so
`setup_metadata.json` records the GPU engine.

### Environment setup

**Option A — Conda**

```bash
conda create -n em_workshop python=3.10
conda activate em_workshop
```

**Option B — Python venv**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install --upgrade pip
```

### Install dependencies

With your environment activated, install the required Python packages:

```bash
pip install voila ipywidgets plotly numpy ipykernel matplotlib scipy segyio joblib
```

| Package     | Purpose |
|-------------|---------|
| voila       | Serves notebooks as standalone web apps (code hidden) |
| ipywidgets  | Interactive widgets in the GUI |
| plotly      | Interactive plots |
| numpy       | Numerical arrays |
| ipykernel   | Jupyter kernel for Python (required for Voila) |
| matplotlib  | Plotting (used by workshop modules) |
| scipy       | Scientific computing (used by workshop modules) |
| segyio      | Read/write SEG-Y files (required by 01_fw_setup) |
| joblib      | Parallelizes the 1D inversion's Tx x seed ensemble across CPU cores (05) |

**Conda users:** After installing, register the kernel so Voila can find it:

```bash
python -m ipykernel install --user --name=em_workshop --display-name="Python (em_workshop)"
```

## 2) Get the workshop

```bash
git clone git@github.com:wiktorwe/EM_inversion_workshop_private.git
cd EM_inversion_workshop_private
```

## 3) Configure and run the workshop GUIs

Launch each step from the project root:

```bash
./start_00_configure.sh
./start_01_fw_setup.sh
./start_02_fwmodelling_and_data_visualization.sh
./start_03_2d_inversion.sh
./start_04_2d_inversion_results.sh
./start_05_1d_inversion.sh
./start_06_1d_inversion_results.sh
```

**Step 00** writes `workshop_config.json` (rockem-suite path, `mpirun`, default SEG-Y, optional GPU flags for 2D forward/inversion, etc.). Run it once on each machine before the other steps.

Each command starts a Voila app for that stage. Steps 05/06 are an independent 1D layered inversion against rockem-suite's analytic magnetic line-source solver — run them after 01/02 have produced FD data for at least one Tx gather.

### Restore a pristine workspace

To remove all generated artifacts (forward models, inversion runs, results):

```bash
./clean.sh          # interactive
./clean.sh -y       # skip confirmation
./clean.sh --dry-run
```

This removes `workspace/` plus Voila PID files and caches. It does not delete notebooks, scripts, or the example model in `examples/`.

## 4) Workspace layout

All notebook outputs go under `workspace/` (gitignored):

```
workspace/
  2D/
    forward/              # FD model, mod.cfg, shot gathers (from step 01–02)
    inversion/
      input/              # prepared 2D inversion inputs (step 03)
      Run{N}/             # 2D inversion runs (step 03)
    results/Run{N}/       # SEG-Y exports from step 04
  1D/
    inversion/Run{N}/     # 1D inversion runs (step 05)
    results/Run{N}/       # SEG-Y exports from step 06
```

Example resistivity model: `examples/Fault_1.sgy` (load in step 01).

## 5) Recommended workflow

0. **Step 00 — Configure**
   - Set rockem-suite path, MPI launcher, and default SEG-Y.
   - Optionally enable **GPU** 2D forward (`mpiEmmodTE2dGpu`) and/or inversion (`mpiEminvTE2dGpu`); click **Validate** to check binaries and `nvidia-smi`.
   - Click **Save** (writes `workshop_config.json`).

1. **Step 01 — FW setup**
   - Load a SEG-Y resistivity model (default: `examples/Fault_1.sgy`).
   - Configure source/survey settings.
   - Click **Generate FD inputs (Finalize setup)**.
   - Creates `workspace/2D/forward/`.

2. **Step 02 — FW modelling and data visualization**
   - Run forward modelling.
   - Inspect Hx/Hz data, amplitudes, and phases.

3. **Step 03 — 2D inversion**
   - Generate inversion inputs.
   - Start inversion and monitor progress.

4. **Step 04 — 2D inversion results**
   - Compare models and data.
   - Export outputs as needed.

5. **Steps 05–06 — 1D inversion and results**
   - Run 1D layered inversion on FD observations.
   - Review and export pseudo-2D sections.

## 6) Documentation

- Full GUI guide with parameter descriptions: [`doc/gui_manual.pdf`](doc/gui_manual.pdf) (build from [`doc/gui_manual.tex`](doc/gui_manual.tex) — see [`doc/README.md`](doc/README.md))
- Example model: [`examples/Fault_1.sgy`](examples/Fault_1.sgy)
- Module reference: [`scripts/README.md`](scripts/README.md)

## 7) Troubleshooting

- If a step reports missing setup data, run Step 01 first and finalize setup.
- If modelling cannot start, verify:
  - `mpirun` is available in your shell
  - `$ROCKEM_SUITE_ROOT/bin/mpiEmmodTE2d` exists (or `mpiEmmodTE2dGpu` if GPU forward is enabled in Step 00) — build CPU with `make mpi`, GPU with `make -f Makefile.gpu bin/mpiEmmodTE2dGpu`
- If inversion cannot start, verify:
  - `mpirun` is available in your shell
  - `$ROCKEM_SUITE_ROOT/bin/mpiEminvTE2d` exists (or `mpiEminvTE2dGpu` if GPU inversion is enabled)
- If a GUI does not open, ensure `voila` is installed in the active environment.
- If you see "No Jupyter kernel for language 'python' found", install `ipykernel` and register the kernel (see Install dependencies above).
