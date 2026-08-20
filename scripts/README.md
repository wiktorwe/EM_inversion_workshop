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
- `fdtd_analytic_calibration.py`: global FDTD–analytic scale `C(f)` for
  Steps 05/06. Notebook 02 offers two Earth models (last successful run wins
  in `setup_metadata.json`): homogeneous `rho_min` (receivers at ±depth so Hz
  enters `C`), or 1D lateral average of production `sg.rss` with Step 01
  survey offsets / apertx. Homogeneous path still uses a purpose-sized
  source-centred domain; lateral-average reuses production `rz0−tz0`.
- `analytic_1d_forward.py`: 1D layered forward model for the workshop's
  magnetic (Hx) line source, via rockem-suite's validated
  `magnetic_line_source_fields_layered` - used by `05_1d_inversion`'s
  inversion and by FDTD–analytic calibration. Replaces `empymod_1d_forward.py`
  (below).
- `inversion_tuning.py`: single-Tx DE budget and Tikhonov λ L-curve helpers
  used by Step 05 (parallel over seeds / λ values on the QC transmitter).
- `inversion.py`: 2D inversion input preparation and `inv.cfg` writing
  helpers, used by `03_2d_inversion`.
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

  The script does not re-run modelling or inversion. `--compile` runs
  `pdflatex` if it is on PATH. `./clean.sh` removes the report with `workspace/`.
- `validate_notebooks.py`: smoke-test each notebook's setup cell (`python scripts/validate_notebooks.py --expect-rockem-missing` if rockem-suite is not configured yet).
- `normalize_notebooks.py`: add nbformat cell ids to all workshop notebooks (run after editing `.ipynb` files).
- `../jupyter_config/jupyter_server_config.py`: Voila websocket settings used by `start_*.sh`.

The notebooks import from `scripts.modules.*` and templates in
`scripts/templates` so logic is not tied to temporary project folders.
