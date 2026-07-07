# Workshop Scripts

This folder is the module-oriented script codebase used by the GUI notebooks
(`00_configure_workshop`, `01_fw_setup` through `06_1d_inversion_results`).

## `scripts/modules/`

- `workshop_config.py`: loads/saves `workshop_config.json` at the repo root
  (rockem-suite path, MPI launcher, default SEG-Y, workspace dir). Import via
  `load_config()` before `rockem_bridge`.
- `rockem_bridge.py`: locates the validated `rockem-suite` checkout (default
  `~/software/new_rockem/rockem-suite`, override via `ROCKEM_SUITE_ROOT`), puts
  its `python/` package and the layered Green's-function solver's `shared/`
  directory on `sys.path`, and re-exports `rockem.config`/`model`/`run`/
  `survey`/`utils`/`wavelet`, the analytic solvers (`line_source_fields_layered`,
  `magnetic_line_source_fields_layered`, `GreensSolverError`),
  `steady_state_phasor`, and `binary_path()`. Import this before anything else
  that needs `rockem.*` or the analytic solvers.
- `segy.py`: SEGY read/write helpers, resistivity resampling to a template
  grid, and `pad_resistivity_for_depth_margin` (pads a loaded model in depth
  if it falls short of `design_explicit_fd`'s required source/receiver
  clearance from the domain edge - see `fd.py`).
- `source.py`: wavelet creation helper.
- `survey.py`: survey config and `Survey.rss` helpers.
- `fd.py`: explicit-engine (`mpiEmmodTE2d`/`mpiEminvTE2d`) FD design
  (`design_explicit_fd` - dx/dt/eps_r/safety/PML/aperture/depth-margin),
  RSS interpolation, and `mod.cfg` update helpers.
- `fd_visualization.py`: FD shot-gather loading and steady-state
  phasor-ratio channel-gain extraction (`compute_gains_for_fd_outputs`).
- `analytic_1d_forward.py`: 1D layered forward model for the workshop's
  magnetic (Hx) line source, via rockem-suite's validated
  `magnetic_line_source_fields_layered` - used by `05_1d_inversion`'s
  inversion and calibration. Replaces `empymod_1d_forward.py` (below).
- `inversion.py`: 2D inversion input preparation and `inv.cfg` writing
  helpers, used by `03_2d_inversion`.
- `empymod_1d_forward.py`: **legacy** - the pre-redesign empymod-based 1D
  point-dipole forward. No longer imported by any of the six workshop
  notebooks; kept only because the vendored `third_party/empy_blockinv`
  example scripts still reference it. Use `analytic_1d_forward.py` instead
  for anything workshop-related.

## `scripts/templates/`

- `survey.cfg`: survey template copied to a temp workspace.
- `mod.cfg`: 2D forward-modelling config template (explicit TE2D engine).
- `mod3d.cfg`: 3D forward-modelling config template (legacy ADI path -
  unvalidated by the 2D redesign, see `02_fwmodelling_and_data_visualization`'s markdown).
- `inv.cfg`: 2D inversion config template (explicit TE2D engine).
- `runmod.sh` / `runinv.sh`: run scripts invoking `mpiEmmodTE2d`/
  `mpiEminvTE2d` at `$ROCKEM_SUITE_ROOT/bin` (default
  `~/software/new_rockem/rockem-suite`, override via the `ROCKEM_SUITE_ROOT`
  environment variable).
- `clean.sh`: removes generated FD/inversion outputs from a run directory.

Other utilities:

- `../clean.sh` / `clean_workspace.py`: remove the entire `workspace/` tree and restore a pristine checkout.

The notebooks import from `scripts.modules.*` and templates in
`scripts/templates` so logic is not tied to temporary project folders.
