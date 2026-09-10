# Workshop Scripts

This folder is the module-oriented script codebase used by the GUI notebooks
(`00_configure_workshop`, `01_fw_setup` through `06_1d_inversion_results`).

## `scripts/modules/`

- `workshop_config.py`: loads/saves `workshop_config.json` at the repo root
  (rockem-suite path, MPI launcher, default SEG-Y, workspace dir, optional GPU
  flags for 2D TE2D forward/inversion). Import via `load_config()` before
  `rockem_bridge`. Key helpers:
  - `forward_engine_te2d()` / `inversion_engine_te2d()` — CPU or GPU binary name
  - `validate_config()` — checks CPU binaries (required); GPU binaries and
    `nvidia-smi` are informational when GPU flags are off, and required
    (with an untick hint) when they are on
  - `patch_runinv_template()` — inject MPI launcher and inversion binary into
    `runinv.sh` (used by Step 03)
- `setup_defaults.py`: read shared defaults from
  `workspace/2D/forward/setup_metadata.json` (`flist_hz`, `n_periods_extract`,
  `eps_r_used`, rho bounds). Notebooks 02/04/05/06 must use this instead of
  hardcoding frequencies.
- `run_report.py`: build/write per-run `analytic_1d_inversion_summary.json` and
  human-readable `REPORT.md` for Step 05; HTML panel helper for Step 06.
- `rss_model.py`: read 2D RSS conductivity models as ``(x, z, grid)`` using the
  same axis convention as notebooks 04/06.
- `report_figures.py`: matplotlib PDF writers used by the workflow report
  (forward model, wavelet, modelled amp/phase, calibration, 2D/1D results).
- `workshop_report.py`: discover workspace artifacts, collect settings tables,
  write `workspace/report/workflow_report.tex`. Invoked by
  `scripts/make_workshop_report.py`.
- `rockem_bridge.py`: locates the validated `rockem-suite` checkout (default
  `~/software/new_rockem/rockem-suite`, override via `ROCKEM_SUITE_ROOT`), puts
  its `python/` package and the validation examples' `shared/` directory on
  `sys.path`, and re-exports `rockem.config`/`model`/`run`/
  `survey`/`utils`/`wavelet`, the analytic solvers from `rockem.greens`
  (`line_source_fields_layered`, `magnetic_line_source_fields_layered`,
  `magnetic_z_line_source_fields_layered`,
  `tilted_magnetic_line_source_fields_layered`, `project_tilted_h`,
  `GreensSolverError`), `steady_state_phasor`, and `binary_path()`. Import this
  before anything else that needs `rockem.*` or the analytic solvers.
  The Green's solvers are package code as of rockem-suite `6723d49`; they used
  to be example-local files under
  `doc/examples/validate_layered_1d_model/shared/`. `phasor.py` did not move,
  which is why that directory is still on `sys.path`. `rockem.greens` imports
  `scipy` (`scipy.special.hankel2`) at module level.
- `segy.py`: SEGY read/write helpers, resistivity resampling to a template
  grid, and `pad_resistivity_for_depth_margin` (pads a loaded model in depth
  if it falls short of `design_explicit_fd`'s required source/receiver
  clearance from the domain edge - see `fd.py`).
- `source.py`: wavelet creation helper.
- `survey.py`: survey config and `Survey.rss` helpers.
- `fd.py`: explicit-engine (`mpiEmmodTE2d`/`mpiEminvTE2d`) FD design
  (`design_explicit_fd` - dx/dt/eps_r/safety/PML/aperture/depth-margin),
  sizing `dt` via `rockem.utils.explicit_em_cfl_dt` (order-aware, σ-independent
  Yee CFL matching the engine's `checkStability`), RSS interpolation, and
  `mod.cfg` update helpers.
- `fd_visualization.py`: FD shot-gather loading and steady-state
  phasor-ratio channel-gain extraction (`compute_gains_for_fd_outputs`).
  Windows the trace and the injected wavelet on the same ABSOLUTE time interval
  and corrects the residual sub-sample offset exactly. This is not cosmetic:
  production records are written at `dtrec` (1e-5 s) while the wavelet is written
  at the model `dt` (2.5e-8 s), so "the last N samples" of each start up to one
  `dtrec` apart - which put a frequency-proportional phase error of up to 21.6
  degrees into every production gain, plus ~173 degrees of period-quantisation
  aliasing at 6 kHz. `C(f)` could not absorb it because the calibration runs
  write `dtrec = dt_model`, where the offset is identically zero. Fixing it took
  the true model's chi-squared from 381.8 to 0.028. See
  `doc/numerics_findings.md` section 2b.
- `fdtd_analytic_calibration.py`: global FDTD–analytic scale `C(f)` for
  Steps 05/06. Notebook 02 offers two Earth models (last successful run wins
  in `setup_metadata.json`): homogeneous `rho_min` (receivers at ±depth so Hz
  enters `C`), or 1D lateral average of production `sg.rss` with Step 01
  survey offsets / apertx. Homogeneous path still uses a purpose-sized
  source-centred domain; lateral-average reuses production `rz0−tz0`.
- `analytic_1d_forward.py`: 1D layered forward model for the workshop's
  magnetic line source (Kx or Kz), via rockem-suite's validated
  `magnetic_line_source_fields_layered` - used by `05_1d_inversion`'s
  inversion and by FDTD–analytic calibration. Replaces `empymod_1d_forward.py`
  (below).
- `inversion_tuning.py`: single-Tx DE budget and Tikhonov λ L-curve helpers
  used by Step 05 (parallel over seeds / λ values on the QC transmitter). Its
  model parameterisation, bounds and per-Tx forward now come from
  `inversion_1d` rather than being a third verbatim copy - that copy had gone
  stale (it still forced every receiver to the first receiver's depth after the
  notebook was fixed), which is the concrete cost of duplicating a misfit.
- `inversion.py`: 2D inversion input preparation and `inv.cfg` writing
  helpers, used by `03_2d_inversion`. `prepare_inversion_inputs` pins
  `order`/`lpml`/`pml_*` from the forward run's `mod.cfg`, so the FWI's
  re-modelled wavefield can never drift from the data it is fitting. (The
  `apertx = "60"` in `templates/inv.cfg` is only a template fallback: notebook
  03's widget defaults to the forward run's own `apertx_m`, 110.6 m for the
  shipped survey.)
- `inversion_1d.py`: the 1D layered model parameterisation and complex-gain
  misfit (`unpack_model_params`, `forward_analytic_for_tx`,
  `complex_gain_objective`, `build_bounds`). These used to live inside
  `05_1d_inversion.ipynb`'s single code cell, which made them unreachable from
  any script; the notebook now imports them. `complex_gain_objective` takes a
  `freq_mask`, which is what makes one stage of a multi-scale frequency ladder
  a slice rather than a rewrite - per-frequency `sigma_hx`/`sigma_hz`/`C` are
  sliced with the same mask, so each stage gets that frequency's own
  calibration scatter instead of a band-averaged one.
- `multiscale_2d.py`: the frequency-ladder machinery for 2D FWI over Task 2's
  per-frequency grids - the grid handoff (`resample_model_log_rho`, which
  interpolates in **log resistivity** and writes an inspectable `.rss`;
  `verify_roundtrip`, which checks the operator on the known true model first)
  and the schedules (`stage_knot_spacing` scales the B-spline `dtx`/`dtz` with
  skin depth; `stage_regularisation` relaxes the Tikhonov weight as frequency
  rises, anchored so the final stage matches the single-stage baseline).
- `headless.py`: non-GUI drivers for the Step 01/02 pipeline
  (`SetupParams`, `build_forward_inputs`, `build_per_frequency_forward_inputs`,
  `run_forward`, `run_calibration`). Every numerical decision is delegated to
  the same helpers the notebooks call - this is a driver, not a second
  implementation. Verified to reproduce notebook 01's `sg.rss`/`ep.rss`/
  `wav2d.rss`/`Survey.rss` byte-for-byte. Used by everything under
  `scripts/experiments/`.
- `empymod_1d_forward.py`: **legacy** - the pre-redesign empymod-based 1D
  point-dipole forward. No longer imported by any of the six workshop
  notebooks; kept only because the vendored `third_party/empy_blockinv`
  example scripts still reference it. Use `analytic_1d_forward.py` instead
  for anything workshop-related. Two things to know before reusing it:
  empymod's `ab` code is RECEIVER-then-SOURCE (`ab=64` is Hz-from-Hx, not
  `46`) - this module had it transposed until the `rockem.greens` migration;
  and `_forward_component` still has an unfixed shape bug for `nfreq == 1`
  with several receivers. Both are documented in the module docstring.

## `scripts/templates/`

- `survey.cfg`: survey template copied to a temp workspace.
- `mod.cfg`: 2D forward-modelling config template (explicit TE2D engine).
- `mod3d.cfg`: 3D forward-modelling config template (legacy ADI path -
  unvalidated by the 2D redesign, see `02_fwmodelling_and_data_visualization`'s markdown).
- `inv.cfg`: 2D inversion config template (explicit TE2D engine).
- `runmod.sh` / `runinv.sh`: run scripts invoking the explicit TE2D engine at
  `$ROCKEM_SUITE_ROOT/bin` (CPU: `mpiEmmodTE2d` / `mpiEminvTE2d`; GPU:
  `mpiEmmodTE2dGpu` / `mpiEminvTE2dGpu` when enabled in Step 00). `runmod.sh`
  reads the forward engine from `setup_metadata.json`; Step 03 patches
  `runinv.sh` from the template via `patch_runinv_template()`.
- `clean.sh`: removes generated FD/inversion outputs from a run directory.

## `scripts/experiments/`

Measurement scripts. Each one states a hypothesis and reports the number that
confirms or falsifies it; none of them are imported by the notebooks.

- `order_refinement.py`: separates a CONSISTENCY error from a DISCRETISATION
  error by measuring `C(f)` at two grid spacings for two stencil orders. A
  consistency error does not converge under refinement; a truncation error falls
  like `dx^2`. This is the evidence behind `order = "6"`.
- `production_timing.py`: measured (not estimated) wall-clock cost of the
  order-2 -> order-6 change on a full 30-transmitter production run.
- `per_frequency_cost.py`: recomputes the per-frequency vs broadband design cost
  from `design_explicit_fd` rather than trusting a table, with and without the
  `eps_r` cap.
- `per_frequency_production.py`: runs the full 30-transmitter survey once per
  frequency and MEASURES the saving `per_frequency_cost.py` only predicts. It
  leaves the four datasets under `workspace/2D/per_frequency/` in place, which
  is what the per-frequency calibration and the multi-scale ladder consume.
- `per_frequency_ab.py`: runs the calibration once per frequency on that
  frequency's own grid and checks it reproduces the broadband table; also tests
  the phase reference (analysis-window shift invariance) and measures the
  per-frequency interface quantisation.
- `hz_source.py` / `fault_couplings.py`: the Kz (`source_type=5`) source -
  calibration against `magnetic_z_line_source_fields_layered`, the Lorentz
  reciprocity check on the layered Earth, and the 2x2 coupling matrix plus
  cross-term null across the fault. `fault_couplings` warns when the farthest
  transmitters show no variation at all, which means the target lies outside
  `apertx/2` and any "detection distance" is the aperture rather than physics.
- `lookahead.py`: the corrected look-ahead measurement - `apertx` sized from the
  look-ahead range instead of the survey offsets, an extended transmitter line,
  per-frequency grids, and a background taken from a SECOND laterally invariant
  model (`examples/Fault_1_nofault.sgy`) rather than from far transmitters,
  which would be circular. Writes a per-frequency figure of all four couplings
  and the cross-term sum vs tool position; `--plot-only` re-draws it from the
  saved JSON without re-running any FDTD.
- `multiscale_1d.py`: the frequency-ladder schedule prototyped in 1D (seconds
  per inversion) before spending 2D FWI time on it. Judges strategies on data
  misfit over ALL frequencies and on model error against the known truth.
- `kx_convergence.py`: whether the analytic solver's `kx` quadrature is
  converged on the actual `Fault_1.sgy` resistivity range - testing the
  `lam_max` (truncation) leg, not just `n_nodes`, which is blind to it.
- `eps_r_bias.py`: the artificial-permittivity bias the calibration cannot see,
  measured by evaluating the analytic solver at `eps_r_used` and at a physical
  `eps_r`. This is the measurement behind `eps_r_cap = 5000`.
- `interface_snapping.py`: how far half a cell of interface quantisation moves
  the data, against the analytic error budget - the evidence for and against
  turning `unpack_model_params(snap_dz=...)` on.
- `multiscale_2d_run.py`: driver for the 2D frequency ladder. See
  `KNOWN_ISSUES.md` for why the head-to-head it exists for has not been run.
- `tensor_1d_test.py`: does the 1D inversion work with all four tensor
  components? Checks the TRUE model against the FDTD data first - if the truth
  does not fit, nothing downstream means anything - then inverts from a uniform
  start with and without the cross-couplings, scoring both on all four.
- `blockinv_tensor_test.py`: the same question for the BlockInv optimizer, which
  packs its data and Jacobian into flat real vectors. Also the regression gate
  for that packing: on the Kx pair it must reproduce the pre-generalisation
  result exactly.
- `empymod_sign_check.py`: the empymod/native complex ratio for BOTH sources,
  which is what justifies `_EMPY_LINE_SIGN = -1` and the `ab` codes. It runs a
  CONTROL with a deliberately wrong `ab` and FAILS if that also passes - every
  correct pairing comes out at the same 1.000064, which is indistinguishable
  from a harness comparing something against itself unless you check.

## `scripts/dev/`

- `bugsweep.py`: parses every notebook and reports names used but never defined,
  and Buttons never bound to a handler. Catches "deleted a widget, left the
  reference" - the single most common bug shipped from this repo.
- `handlersweep.py`: CALLS every `on_*` / `update_*` / `refresh_*` handler in
  every notebook and fails on `NameError`/`AttributeError`.
  `validate_notebooks.py` executes the cell but never clicks anything, so a
  broken callback passes it. Run both. It stubs `Figure.show`, because plotly
  opens a browser tab per figure outside a notebook.
- `chainsweep.py`: builds a synthetic acquisition-matrix workspace and executes
  every notebook against it, failing if any dataset artifact
  (`setup_metadata.json`, `sg.rss`, `wav2d.rss`, `Data/*.rss`, ...) is still
  bound to the forward ROOT. On a matrix workspace the root holds only
  `manifest.json` and dataset subdirectories, so such a path cannot resolve.
  The repo's own `workspace/` is usually single-dataset, where the wrong path
  still exists - which is why the other checks all passed while Step 05 was
  broken for anyone with a real matrix.
- `nbedit.py`: exact-match editor for the notebooks' single large code cells.
  Each edit declares how many occurrences it expects and fails loudly if the
  count is wrong, so a stale edit cannot silently do nothing.

Other utilities:

- `../clean.sh` / `clean_workspace.py`: remove the entire `workspace/` tree and
  restore a pristine checkout. Prints an explicit warning that setup metadata,
  calibration, and 1D `REPORT.md` files are deleted.
- `make_workshop_report.py`: write a LaTeX snapshot of the current workspace
  (`workspace/report/workflow_report.tex` plus PDF figures under
  `workspace/report/figures/`). Requires Step 01 `setup_metadata.json`. 2D and
  1D inversion sections are included only when a `Run{N}` directory exists
  (latest by default). From the workshop root:

  ```bash
  python scripts/make_workshop_report.py
  python scripts/make_workshop_report.py --compile
  python scripts/make_workshop_report.py --2d-run Run1 --1d-run Run0
  python scripts/make_workshop_report.py --no-2d --no-1d
  ```

  One report covers ONE forward dataset. On an acquisition-matrix workspace:

  ```bash
  python scripts/make_workshop_report.py --list-datasets
  python scripts/make_workshop_report.py --dataset f1000Hz_hx
  python scripts/make_workshop_report.py --all-datasets   # report/<dataset>/ each
  ```

  With no `--dataset` it takes the first and says so. `--all-datasets` writes to
  a subdirectory per dataset because the figure basenames are fixed.

  The script does not re-run modelling or inversion. `--compile` runs
  `pdflatex` if it is on PATH. `./clean.sh` removes the report with `workspace/`.
- `run_matrix.py`: model AND calibrate every dataset of the acquisition matrix
  in one command, sequentially. Step 01 already recorded which frequencies and
  sources were selected; nothing downstream should make you re-enter that one
  dataset at a time.

  ```bash
  python scripts/run_matrix.py --dry-run          # show the plan
  python scripts/run_matrix.py                    # model + calibrate everything
  python scripts/run_matrix.py --skip-modelling   # datasets already modelled
  ```

  Each dataset is calibrated with the source it was modelled with, and one
  `--method` is applied to all of them (mixing methods across sources is what
  the tensor inversion refuses). A failure in one dataset is reported and
  stepped over rather than abandoning the batch.
- `validate_notebooks.py`: execute EVERY code cell of every notebook as a smoke
  test (`python scripts/validate_notebooks.py --expect-rockem-missing` if
  rockem-suite is not configured yet). It used to run only the first code cell,
  which left notebook 04's GUI cell - the larger of its two - unchecked. Run it
  in an environment that has the GUI dependencies, or every notebook "fails" for
  the wrong reason.
- `normalize_notebooks.py`: add nbformat cell ids to all workshop notebooks (run after editing `.ipynb` files).
- `../jupyter_config/jupyter_server_config.py`: Voila websocket settings used by `start_*.sh`.

The notebooks import from `scripts.modules.*` and templates in
`scripts/templates` so logic is not tied to temporary project folders.
