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

### Remembered parameters (do not retype them)

- **Step 01** writes `workspace/2D/forward/setup_metadata.json` with frequencies
  (`flist_hz`), `n_periods_extract`, `eps_r_used`, survey geometry, and FD design.
  Later notebooks (02, 04, 05, 06) read those values as widget defaults.
- **Step 02** appends the FDTD–analytic calibration into the same file, as a
  VALIDATION record. `C(f)` is COMPUTED, not fitted - `C = dx*dz*s(order)`, from
  how the engine injects a source into one cell (`scripts/modules/fd_error_model.py`).
  Steps 05/06 compute it and do not read it from `setup_metadata.json`; the
  Step 02 run reports how far the fitted value lands from the computed one
  (measured worst 0.430 %, band 0.5 %).
- **Step 05** writes a full per-run report under each
  `workspace/1D/inversion/Run{N}/` folder:
  `REPORT.md` (human-readable), `analytic_1d_inversion_summary.json`, and
  `run_metadata.json` (optimizer, weights, bounds, frequencies, calibration).
  Step 06 loads those when you open a run — you do not need to remember what
  frequencies or inversion settings were used.

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

This removes `workspace/` plus Voila PID files and caches. It does **not** delete notebooks, scripts, or the example model in `examples/`.

Cleaning **does** delete remembered setup parameters and run reports:

- `setup_metadata.json` (frequencies / design from Step 01)
- the Step 02 validation record stored there (not needed to invert - C is computed)
- every `Run{N}/REPORT.md` and summary from Step 05

After `./clean.sh`, re-run Step 01 before Steps 02-06. Step 02's validation is
optional: `C(f)` is computed, so an uncalibrated workspace still inverts.

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
   - Configure source/survey settings and frequency list.
   - Click **Generate FD inputs (Finalize setup)**.
   - Creates `workspace/2D/forward/` and writes `setup_metadata.json` (remembered by later steps).

2. **Step 02 — FW modelling and data visualization**
   - Frequencies / `n_periods` load from `setup_metadata.json`.
   - Run forward modelling. **Validate C** (homogeneous `rho_min` or 1D lateral
     average of the true model) is optional - it checks the computed `C(f)`
     against a fitted one; the inversion computes `C` either way.
   - Inspect Hx/Hz data, amplitudes, and phases.

3. **Step 03 — 2D inversion**
   - Generate inversion inputs.
   - Start inversion and monitor progress.

4. **Step 04 — 2D inversion results**
   - Compare models and data (freqs / `n_periods` from setup metadata).
   - Export outputs as needed.

5. **Steps 05–06 — 1D inversion and results**
   - Step 05 defaults freqs / `n_periods` / rho bounds from setup metadata; each run writes `REPORT.md`.
   - Step 06 loads a run and shows its parameters (freqs, weights, optimizer) from that report.

## 6) Modelling notes and limitations

### Anisotropy: TE2D takes no anisotropy input, and correctly needs none

This workshop is **isotropic**, and on the 2D TE engine it drives that is not a
simplification you can lift — it is structural.

`ModelEmTE2D`'s constructor is `(Sgfile, Epfile, lpml)`: there is no anisotropy
argument. That is correct rather than an omission. TE2D's only electric-field
component is `Ey`, which is **horizontal**, so the field only ever sees the
horizontal material properties `Asg*Sg` and `Aep*Ep`. A VTI ratio is therefore
exactly degenerate with `Sg` and `Ep` themselves: any anisotropic model produces
a field identical to some isotropic one, so anisotropy is **unresolvable** in
this engine, not merely unmodelled. `rockem.config.write_te2d_config` documents
the same thing.

The practical consequence: **VTI is not out of scope for this workshop, it is
unrepresentable in the engine the workshop uses.** Anyone who needs it must move
to `TM2D` (`Ex, Ez, Hy` — the extraordinary branch, which does see the vertical
properties) or to 3D. Do not spend time trying to add an anisotropy file to the
2D TE path.

For the engines that *do* take anisotropy (`ModelEmTM2D`, `ModelEm3D`),
rockem-suite's model is **two independent optional ratio fields**, not the single
legacy key:

| key   | meaning                                    |
|-------|--------------------------------------------|
| `Aep` | `eps_h / eps_v` — dielectric anisotropy     |
| `Asg` | `sigma_h / sigma_v` — conductivity anisotropy |

Both are optional: an omitted key (or an empty filename) makes rockem synthesise
an all-ones array, so **an isotropic model ships no anisotropy file at all**.
This workshop therefore writes none, for any dimension.

Two traps worth stating plainly:

- The old single `A` key (which applied to permittivity *and* conductivity) is
  now **rejected with a fatal error**, not ignored. The workshop's `mod3d.cfg`
  used to set it, which made the 3D path fail outright on a current
  rockem-suite; that key is gone.
- **`Asg` is a conductivity ratio**, i.e. the *reciprocal* of the geophysical
  coefficient of anisotropy `rho_h / rho_v`. This is the easiest thing in the
  codebase to get upside down coming from a resistivity workflow.

### Two magnetic source components: Kx and Kz

The workshop used to hardcode a single **Kx** magnetic line source
(`source_type = "3"`). Both receiver components were already recorded, so a
second run with **Kz** (`source_type = "5"`) completes the full 2×2 magnetic
coupling matrix:

| | records Hx | records Hz |
|---|---|---|
| **Kx source** | `Cxx` (coaxial, strong) | `Cxz` (cross) |
| **Kz source** | `Czx` (cross) | `Czz` (transverse, strong) |

The source component is now a parameter everywhere — a **cal source** dropdown
in Step 02, `source_field` in `scripts.modules.headless.SetupParams`, and an
argument on every calibration entry point. Each source gets its own run
directory and its own fitted `C(f)`; both are stored under
`fdtd_analytic_calibration_by_source` in `setup_metadata.json`, while the
*active* `C` that notebooks 05/06 consume stays the **Kx** one (the 1D
inversion's forward model is a Kx line source, so handing it a Kz `C` would
silently calibrate the wrong source).

Two reasons this is worth the extra run:

1. **Any tilt, for free.** A tilted magnetic line source is an exact
   superposition `cos(θ)·Kx + sin(θ)·Kz`, and a tilted receiver coil is
   `cos(θ)·Hx + sin(θ)·Hz` — see `tilted_magnetic_line_source_fields_layered`
   and `project_tilted_h`. Two FDTD runs therefore buy the entire in-plane tilt
   space at zero marginal cost per configuration. (Out-of-plane tilt is **not**
   representable: a Ky moment drives the TM mode, a different field triple
   entirely.)
2. **A null background to look for structure against.** At the survey's zero
   depth offset, Lorentz reciprocity plus the odd-in-offset parity of the cross
   components forces

   ```
   Hz(Kx@p)  =  −Hx(Kz@p)      for the same source position and the same receiver
   ```

   in **any** laterally-invariant Earth (verified to 5e-16 on a 3-layer stack;
   the identity does *not* hold once the receiver is at a different depth, so it
   is specific to this colinear geometry). Their sum `S = Cxz + Czx` is
   therefore identically zero in the background model and non-zero **only** where
   lateral structure breaks the symmetry — the fault appears against nothing
   rather than as a small perturbation on a large direct coupling.
   `scripts/experiments/fault_couplings.py` measures the tool-to-fault distance
   at which each observable leaves its background by more than the calibration
   scatter.

### The acquisition matrix: one run per (frequency, source component)

Step 01 builds a **matrix** of forward datasets. There is no broadband run and
no single-dataset option anywhere in the workshop:

- **sources** — a multi-select (Kx and/or Kz). Selecting both builds both
  datasets and so completes the 2×2 magnetic coupling matrix.
- **one dataset per frequency, always.** Spatial sampling is set by the
  *highest* frequency and record length by the *lowest*, so a broadband run
  applies the fine grid of the top tone through the long record of the bottom
  one and pays for both. Measured on this survey: **312.1 s split against
  621.9 s broadband** — 1.99× serially, up to 5.84× concurrently. Each dataset
  gets its own `dx`/`dt`/`eps_r`, its own **single-tone wavelet**, and its own
  calibration.

With the shipped defaults that is 4 frequencies × 2 sources = **8 datasets**,
written as subdirectories of `workspace/2D/forward/` with a `manifest.json`
listing them.

**Whatever you select in Step 01 runs together from then on, and there is no
way to run less than all of it.** Step 02 models, extracts and calibrates every
dataset in one action each; Step 03 stages and runs every dataset's inversion
sequentially; Step 05 fits every frequency and every available tensor component
in one inversion; `make_workshop_report.py --all-datasets` writes a report per
dataset. Headless, the whole matrix is one command:

```bash
python scripts/run_matrix.py            # model + calibrate every dataset
```

The per-dataset **Run modelling**, **Calibrate (homogeneous)** and **Calibrate
(lateral average)** buttons, the **cal source** dropdown and the Step 05
component selector have all been removed. The `dataset` dropdowns that remain
change only *what you look at*, never what runs.

The 2D inversion runs one JOINT inversion per frequency: `mpiEminvTE2d` takes a
comma-separated `source_type`, so its work list spans (shot × source type) and
the Kx and Kz gradients are summed before the model is stepped. A 4-component
acquisition is therefore one inversion whose every model update sees all four
components, not two runs where the second overwrites the first's answer. The
sequence over frequencies is one action over the whole matrix, not a choice you
make dataset by dataset.

The historical layout is preserved exactly: one broadband run with a single
source still writes straight into `workspace/2D/forward/` with
`setup_metadata.json` where it has always been, so the change can be A/B tested
and older workspaces keep loading.

Step 05 consumes the **whole** matrix: every per-frequency dataset contributes
the one tone it was designed for, both sources contribute their two tensor
components, and the inversion fits them together. Step 03 consumes it a
frequency at a time, both sources at once. Steps 04 and 06 look at one result at
a time by nature.

Each per-frequency dataset has its own grid and so its own calibration, and
Step 05 assembles C per frequency to match. Worth knowing when reading those
numbers: `|C| ∝ dx²`, so raw `|C|` is **not** comparable between datasets —
measured 2.61 / 1.95 / 0.90 / 0.64 at 1/2/4/6 kHz purely because dx is
1.6 / 1.4 / 0.95 / 0.8 m. Normalised as `C/dx²` the genuine spread is 2.04 % at
1 kHz, and the 6 kHz same-grid control reproduces the broadband value to
0.001 %.

Inverting the full tensor jointly in 2D is what Step 03 now does.
`source_type = "3,5"` makes the run joint, and the four observed gathers are
named per (source, receiver): `Recordfile_HX_HZ` is Hz recorded from a Kx
source. Two things follow. Every record file must share a trace count, trace
order, per-trace coordinates and time axis — the engine builds ONE shot keymap
from the first and `apertx > 0` is a source-centred total width, so it checks
this at startup and aborts naming the offending file and trace index. And the
cost is about what the separate single-source runs cost combined, roughly 2×
one source rather than 4×, because both receiver components already come from
one propagation per (shot, source).

The acceptance test is `scripts/experiments/joint_source_gradient.py`: on a tiny
dedicated model it checks that the joint gradient equals the sum of the two
single-source gradients (measured **4.4e-08** relative L2, against a 1e-05
tolerance — round-off, because `Image2D::stackImage` accumulates in float), that
the misfit totals agree exactly, that single-source runs are untouched, and that
`t(joint) / (t(Kx) + t(Kz))` stays under 1.25 (measured 0.95-0.98).

The same identity was checked once at production scale, off-script: 2 kHz,
30 shots, `order = 6`, `lpml = 13`, `apertx = 110.6 m`, `constrain = true`,
`max_iterations = 0`, `-np 6`. Relative L2 **1.01e-07**, misfit totals equal to
the last digit, and `misfit.rss` 30 + 30 -> **60** entries. The identity does
not depend on shot count, so the committed test uses the tiny model.

### Extraction window must exclude the source ramp-up

`n_periods_extract` in `setup_metadata.json` is **not** the wavelet's own
`n_periods`, and this matters more than it looks.

The source is a ramped CW wavelet (`Ramp_sqw`), whose ramp lasts `alpha / f_min`
seconds — half a period of the lowest tone at the shipped `alpha = 0.5`.
`steady_state_phasor` analyses only the **last** `n_periods` periods precisely so
that this startup transient is excluded. Asking for `n_periods_extract` equal to
the wavelet's `n_periods` asks for the *whole record*, which defeats that
protection: the ramp leaks into every phasor, frequency-dependently, and shows up
as an apparent physics drift.

Measured on the workshop's own calibration (order 6, homogeneous Earth,
1–6 kHz): `|C|/dx²` drifts **0.230 %** across the band with the full-record
window and **0.023 %** once the ramp is excluded — a factor of ten. A
non-integer window is worse still (0.538 % at 4.5 periods), because it holds a
non-integer number of periods of some tones and leaks between them.

Step 01 now writes the largest safe *integer* window
(`n_periods_extract_safe = n_periods − ceil(alpha)`, i.e. 4 for the shipped
defaults), and `steady_state_gains` warns if it is ever asked for a window as
long as the record.

### Channel gains are windowed on absolute time, not "the last N samples"

The production shot record is written at `dtrec` (1×10⁻⁵ s by default) while the
injected wavelet is written at the model `dt` (2.5×10⁻⁸ s). Their records are
therefore **different lengths on different sample grids**, so taking "the last
N samples" of each — which is what `steady_state_phasor` does unaided — makes
the two analysis windows start up to one `dtrec` apart in absolute time.

That is a pure time shift, so it appears as a phase error **proportional to
frequency**, and it does *not* cancel in the trace/wavelet ratio. Critically,
`C(f)` cannot absorb it either: the calibration runs write
`dtrec = dt_model`, where trace and wavelet have identical length and the offset
is exactly zero. The bug therefore lived only on the path from production data
to the 1D inversion, and every check that might have caught it ran on the
geometry where it does not exist.

Measured before the fix, on a transmitter far from the fault where the Earth is
essentially 1D: −3.5° / −7.3° / −14.2° at 1 / 2 / 4 kHz, plus a further ~173° at
6 kHz from `round(1/f/dt)` giving 17 samples per period instead of 16.67. The
**true** model scored a chi-squared of 381.8 — worse than the models the
inversion was finding.

`steady_state_gains` now windows both series on the same absolute interval and
applies the exact residual sub-sample correction. The true model's chi-squared
went 381.8 → 0.028, and `C(f)` is unchanged to five digits. Any 1D inversion run
produced before this is not comparable with one produced after; the 2D FWI is
unaffected, since it fits time-domain traces and never goes through the phasor
extraction.

### The 3D path is legacy and unvalidated

`mod3d.cfg` / `mpiEmmodADI3d` predate the 2D redesign and are not covered by any
of the validation in this workshop. Its config keys are now correct (it no
longer errors on startup), but no Green's-function check has been run against
it here. Treat 2D TE as the supported path.

### Finite-difference stencil order

`order` in `mod.cfg` / `inv.cfg` is **6**, not 2. rock-em's tabulated staggered
first-derivative coefficients are dispersion-optimised (Holberg-type), not
Taylor-exact, and at order 2 they violate the consistency condition
`sum_n c_n (2n+1) = 1` by **−2.3 %** — so every first derivative is under-scaled
by that amount *regardless of dx*, an error that does not converge away under
grid refinement. See the comment block at the top of
[`scripts/templates/mod.cfg`](scripts/templates/mod.cfg) for the measurements,
and `scripts/experiments/` for the scripts that produced them.

## 7) Documentation

- Full GUI guide with parameter descriptions: [`doc/gui_manual.pdf`](doc/gui_manual.pdf) (build from [`doc/gui_manual.tex`](doc/gui_manual.tex) — see [`doc/README.md`](doc/README.md))
- Example model: [`examples/Fault_1.sgy`](examples/Fault_1.sgy)
- Module reference: [`scripts/README.md`](scripts/README.md)

## 8) Troubleshooting

- If a step reports missing setup data, run Step 01 first and finalize setup.
- If modelling cannot start, verify:
  - `mpirun` is available in your shell
  - `$ROCKEM_SUITE_ROOT/bin/mpiEmmodTE2d` exists (or `mpiEmmodTE2dGpu` if GPU forward is enabled in Step 00) — build CPU with `make mpi`, GPU with `make -f Makefile.gpu bin/mpiEmmodTE2dGpu`
- If inversion cannot start, verify:
  - `mpirun` is available in your shell
  - `$ROCKEM_SUITE_ROOT/bin/mpiEminvTE2d` exists (or `mpiEminvTE2dGpu` if GPU inversion is enabled)
- If a GUI does not open, ensure `voila` is installed in the active environment.
- If you see "No Jupyter kernel for language 'python' found", install `ipykernel` and register the kernel (see Install dependencies above).
