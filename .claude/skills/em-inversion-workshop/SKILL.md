---
name: em-inversion-workshop
description: Use when working on the EM_inversion_workshop repo - the Voila/Jupyter workshop for 2D electromagnetic forward modelling and inversion that drives rockem-suite's explicit TE2D engine (mpiEmmodTE2d/mpiEminvTE2d). Covers the six GUI notebooks, scripts/modules, the FD design chain, the FDTD-vs-analytic calibration C(f), the 1D layered inversion, the acquisition matrix (per-frequency x per-source), and the measured numerical findings. Triggers on "workshop", "01_fw_setup", "setup_metadata.json", "mod.cfg", "inv.cfg", "calibration", "C(f)", "1D inversion", "channel gain", "steady_state_gains", "per-frequency", "Kz source", "look-ahead", "Fault_1.sgy".
---

# EM inversion workshop

## RULE 1 - CHANGE THE WHOLE CHAIN, OR DO NOT CHANGE IT

**When you change anything that something else depends on, you change every
consumer in the same edit, and you verify the chain end to end before you stop.**
A change that leaves any downstream step broken is not a partial success. It is
a regression, and it is worse than not having started, because the breakage
surfaces later and somewhere else.

This is the single most important rule in this repo, because the workshop *is* a
chain: Step 01 writes inputs -> Step 02 models and calibrates -> Steps 03/04
invert and display in 2D -> Steps 05/06 invert and display in 1D. Every step
consumes the previous one's files and metadata schema.

Real breakages that happened here, each of which should have been caught in the
same edit that caused it:

| Change | What it broke | How it should have been caught |
|---|---|---|
| Removed `an_path` from `write_sg_ep_rss` | `headless.build_forward_inputs` still passed it - `TypeError` mid-run | grep every caller BEFORE editing the signature |
| Gave `check_kx_convergence` a new (lam_max) leg | kept the old `rel_tol=1e-4`, so Step 05's panel said "NOT CONVERGED" always | re-run the consumer, look at what the user sees |
| Added Kz to the calibration | `forward_1d_gains` stayed hardcoded to Kx, so the 1D inversion could not use it | trace the new capability to every place it must appear |
| Made Step 01 emit an acquisition matrix | Step 02 could only read a single dataset | run the next step, not just the one you edited |
| Added `source_field` to `forward_1d_gains` | the empymod fallback still asked for `ab=44/64`, so a rejected Kz solve silently returned the **Kx** response | follow the new argument into every branch, including error paths |
| Added the per-run acquisition matrix | `iter_datasets` passed legacy manifest entries through unnormalised, with no `source_field` key for consumers to index | define the contract (`DATASET_KEYS`) and make every producer satisfy it |

### The checklist, before you call any change done

1. `grep -rn "<symbol>" --include="*.py" --include="*.ipynb" . | grep -v __pycache__`
   for every function, argument, config key and metadata field you touched.
   Notebooks are code: they will not show up in an import check.
2. Update **every** consumer in the same change.
3. Run `python scripts/validate_notebooks.py` - it executes EVERY code cell of
   every notebook. (It used to run only the first, which meant notebook 04's
   959-line GUI cell was never checked at all.) Use an environment with the GUI
   dependencies; without them every notebook "fails" for the wrong reason.
4. Run the consumer that a user actually operates, and look at its output. An
   import that succeeds is not a behaviour that is correct.
5. If a numeric default changes, re-derive the threshold that depends on it. A
   pass/fail threshold inherited from a different test is worse than none,
   because it looks like a measurement.

### Duplication is how chains rot

This repo has already produced **three** divergent copies of the same 1D misfit
(notebook 05, `inversion_tuning.py`, and a scratch copy), and the stale ones
silently kept a bug that had been fixed in the notebook. Prefer one
implementation imported everywhere:

- `scripts/modules/inversion_1d.py` - 1D parameterisation, misfit, tensor
  forward, `tensor_objective_parts` (THE decomposed misfit - `tensor_objective`
  and `inversion_tuning.split_objective` are both thin wrappers over it),
  `resolve_tensor_calibration` and `component_weights`. The last two used to
  live in notebook 05 where `inversion_tuning` could not reach them, which is
  exactly why the tuners kept a Kx-only copy and tuned a different objective
  from the one being minimised.
- `scripts/modules/headless.py` - the Step 01/02 pipeline as plain functions;
  notebooks call it rather than reimplementing it, and it is verified to
  reproduce notebook 01's outputs byte-for-byte. `iter_datasets` is the one
  dataset enumerator; every entry satisfies `DATASET_KEYS`.
- `analytic_1d_forward.forward_1d_gains` is the ONLY forward. Notebook 05's
  true-model QC overlay had its own solver loop until it was found to be both
  Kx-only and crashing on any survey with more than one receiver.

## RULE 2 - THE DOCS ARE PART OF THE CHAIN

`KNOWN_ISSUES.md` and this skill are consumers like any other. **Any change that
alters behaviour either of them describes must update both in the same edit.**

- Fixed an entry in `KNOWN_ISSUES.md`? Delete it in that change. Do not leave it
  "for later" - the next person will re-investigate a problem that no longer
  exists.
- Found a new limitation while fixing something else? Add it in that same
  change, with what you measured. Most entries in that file were found this way.
- Changed a default, a file name, a metadata key, a function signature or the
  step order? Grep this skill for it and fix the text. The chain map below and
  the `setup_metadata.json` key list are the parts that rot fastest.
- Deleting an entry without the fix is the worst outcome of all: it converts a
  known problem into an unknown one.

Stale documentation in this repo has real cost, because both files are read as
statements of fact about measured behaviour. If a number here cannot be
reproduced by a script in `scripts/experiments/`, it should not be here.

## THE CHAIN, link by link

Read this before editing anything. Each step CONSUMES the artifacts of the
previous one; every arrow is a place a change can break something.

### Step 00 - configure
- **imports** `workshop_config`
- **writes** `workshop_config.json` (repo root, gitignored): rockem-suite path,
  `mpirun`, `nproc`, default SEG-Y, workspace dir, GPU flags
- **consumed by** every other step via `workshop_config.load_config()`. Import it
  BEFORE `rockem_bridge` - it sets `ROCKEM_SUITE_ROOT`.

### Step 01 - forward setup
- **imports** `fd`, `headless`, `segy`, `source`, `survey`, `workshop_config`
- **reads** `examples/*.sgy`, `scripts/templates/{mod.cfg,mod3d.cfg,survey.cfg,runmod.sh,clean.sh}`
- **writes**, per dataset, into `workspace/2D/forward/` (single) or
  `workspace/2D/forward/<freq>_<src>/` (matrix):
  `sg.rss`, `ep.rss`, `wav2d.rss`, `Survey.rss`, `mod.cfg`, `runmod.sh`,
  `clean.sh`, `setup_metadata.json`; plus `manifest.json` at the root
- **key chain facts**
  - `order`/`lpml` come FROM `templates/mod.cfg` via `resolve_fd_order_from_cfg`,
    and `design_explicit_fd` sizes `dt` from `order` via `explicit_em_cfl_dt`.
    Change the template order and dt changes - that is intended, and the engine
    hard-aborts if they disagree.
  - `eps_r_used` goes into `ep.rss` AND into the metadata, because the analytic
    reference must use the same value or the calibration is not comparing like
    with like.
  - the matrix path routes through `headless.build_forward_matrix`; the single
    path is the historical inline code. Both must keep producing identical
    output for the single case - it is the A/B baseline.

### Step 02 - forward modelling + calibration
- **imports** `fd_visualization`, `fdtd_analytic_calibration`, `headless`,
  `rockem_bridge`, `setup_defaults`, `workshop_config`
- **reads** a dataset's `mod.cfg` + inputs, `setup_metadata.json`, `manifest.json`
- **writes** `Data/{Hx,Hz}shot.rss`, `Data/processed/amp_phase_results.npz`,
  `calibration_{homogeneous,lateral_average}[_hz]/`, and the calibration blocks
  `fdtd_analytic_calibration` (active, Kx only) and
  `fdtd_analytic_calibration_by_source` back into `setup_metadata.json`
- **key chain facts**
  - `compute_gains_for_fd_outputs` -> `steady_state_gains` is THE extraction used
    by Steps 02/04/05/06 and the report. Its windowing rules (absolute-time
    alignment, ramp exclusion) are load-bearing for all of them.
  - calibration runs write `dtrec = dt_model`; production runs write
    `dtrec = 1e-5`. That asymmetry is why the extraction must align on absolute
    time.
  - only an HX calibration may take the ACTIVE slot - Steps 05/06 model a Kx
    line source and would otherwise be handed the wrong C.

### Step 03 - 2D inversion staging and run
- **imports** `fd_visualization`, `inversion`, `workshop_config`
- **reads** a forward dataset: `mod.cfg` (for `order`, `lpml`, `pml_*`,
  `source_type`), `sg.rss`, `ep.rss`, `wav2d.rss`, `Data/{Hx,Hz}shot.rss`
- **writes** `workspace/2D/inversion/input/` and `Run{N}/`: `inv.cfg`,
  `sg0.rss`, `ep.rss`, `wav2d.rss`, `Hx_data.rss`, `Hz_data.rss`, `weight.rss`,
  `Sg_min/max.rss`, `Local/`, `Results/`
- **key chain fact** `prepare_inversion_inputs` PINS `order`, `lpml`, `pml_*`
  and `source_type` from the forward `mod.cfg`. Anything the forward run
  chooses that the inversion must match belongs in that list - a mismatch is a
  silently wrong gradient, not an error.

### Step 04 - 2D results
- **imports** `fd_visualization`, `headless`, `rockem_bridge`, `segy`, `setup_defaults`
- **reads** `Run{N}/` models + the forward data + `setup_metadata.json`
  (frequencies, `n_periods_extract`, SEG-Y template geometry)
- **writes** `workspace/2D/results/Run{N}/` SEG-Y exports
- **key chain fact** it has TWO code cells - setup and GUI. `_select_dataset`
  is defined in the first and the dropdown in the second; both run in one
  namespace under Voila. It carries a dataset dropdown built from
  `headless.iter_datasets`, like Step 02.

### Step 05 - 1D layered inversion
- **imports** `analytic_1d_forward`, `fd_visualization`, `fdtd_analytic_calibration`,
  `headless`, `inversion_1d`, `inversion_tuning`, `run_report`, `segy`,
  `setup_defaults`
- **reads** `Data/{Hx,Hz}shot.rss` + `wav2d.rss` of the Kx dataset, the Kz
  dataset if one exists (for the tensor), and the calibration blocks
- **writes** `workspace/1D/inversion/OneDRun{N}/`: `REPORT.md`,
  `analytic_1d_inversion_summary.json`, `run_metadata.json`
- **key chain facts**
  - the forward is `analytic_1d_forward.forward_1d_gains(source_field=...)`;
    it MUST match the source that produced the data.
  - `inversion_1d` holds the one true parameterisation and misfit. Notebook 05
    and `inversion_tuning` both import it - do not re-add a local copy.

### Step 06 - 1D results
- **imports** `analytic_1d_forward`, `fd_visualization`, `fdtd_analytic_calibration`,
  `headless`, `run_report`, `segy`, `setup_defaults`
- **reads** `OneDRun{N}/` summaries (including `run_metadata.json`'s
  `data_convention`) + the forward data
- **writes** `workspace/1D/results/` SEG-Y exports
- **key chain facts**
  - it carries the same dataset dropdown as Steps 02 and 04.
  - it WARNS when a run's `data_convention` is older than
    `run_report.DATA_CONVENTION`. Bump that constant whenever a change makes
    new channel gains incomparable with old ones, and say why in its comment.

### Out of band
`scripts/make_workshop_report.py` -> `workshop_report.py` reads the whole
workspace and writes `workspace/report/workflow_report.tex` + figures. It reads
`setup_metadata.json` and the calibration, so metadata changes reach it too. It
reports on ONE forward dataset: `--dataset NAME`, `--all-datasets` (one report
per dataset under `report/<dataset>/`, because the figure basenames are fixed),
`--list-datasets`. Every report names the dataset it is about.

### `setup_metadata.json` IS the contract

Keys currently read downstream:

```
apertx_m  drx_m  dt_model_target_s  dt_wavelet_s  dtrec_written_s  dtx_m
dx_model_target_m  eps_r_cap_binding  eps_r_used  explicit_cfl_safety  fd_order
fdtd_analytic_calibration  fdtd_analytic_calibration_by_source  flist_hz
forward_cfg  forward_data_dim  forward_engine  forward_wavelet  f_max_hz
f_min_hz  max_offset_m  min_offset_m  n_periods_extract  nrx  ntx  ny_samples
pml_heuristic  rho_max_ohm_m  rho_min_ohm_m  rx0_m  rz0_m  segy_template_path
segy_{ox,oz,dx,dz,nx,nz}  source_field  source_type  tx0_m  tz0_m
wavelet_n_periods  wavelet_ramp_seconds
```

**Adding a key is safe. Changing the meaning of one, or removing one, is a chain
change** - grep every reader first (`setup_defaults.py` and notebooks 02/04/05/06
are the main ones) and update them in the same edit.

Two more contracts sit alongside it:

- `manifest.json` (Step 01, at the forward ROOT) - read only through
  `headless.iter_datasets`, which normalises both manifest schemas so every
  entry has all of `headless.DATASET_KEYS`
  (`name`, `run_dir`, `freq_hz`, `source_field`, `meta`). Index those keys
  freely; anything else, check first.
- `run_metadata.json` (Step 05, per 1D run) - carries `data_convention`, the
  version of the observed-data extraction the run was fitted against. Step 06
  warns when it is older than `run_report.DATA_CONVENTION`.

### Module dependency order

`workshop_config` -> `rockem_bridge` -> everything else. `rockem_bridge` puts
rockem-suite's `python/` on `sys.path`, so nothing that needs `rockem.*` may be
imported before it.

`analytic_1d_forward` <- `inversion_1d` <- {`inversion_tuning`, notebook 05,
`scripts/experiments/{multiscale_1d,tensor_1d_test,blockinv_tensor_test}.py`}
`empymod_line_source` <- `analytic_1d_forward` (contrasted-interface fallback
only; it takes `source_field` and MUST be given it)
`fd` + `segy` + `source` + `survey` <- `headless` <- {notebook 01, notebook 02,
all of `scripts/experiments/`}
`fd_visualization` <- {notebooks 02/04/05/06, `workshop_report`, experiments}

## Things that are true here and cost time when forgotten

- **Stencil order is 6, not 2.** rock-em's tabulated coefficients are
  dispersion-optimised, not Taylor-exact; at order 2 every first derivative is
  under-scaled by 2.3 % *regardless of dx*, and it does NOT converge away under
  refinement. See `scripts/templates/mod.cfg` and `doc/numerics_findings.md`.
- **`dtrec` != model `dt`.** Production runs record at 1e-5 s while the wavelet
  is written at the model dt, so their records differ in length.
  `steady_state_gains` therefore windows both on ABSOLUTE time and corrects the
  sub-sample remainder. Do not "simplify" that back to "the last N samples".
- **`n_periods_extract` must exclude the source ramp** (`alpha/f_min` seconds),
  and must be an integer number of `f_min` periods.
- **`.rss` samples sit at `o + k*d`**, not at cell centres.
- **`eps_r_cap` is 5000, not 1000.** At 1000 it BOUND at 1 and 2 kHz and threw
  away `dt` for nothing; the bias removing it costs was measured at
  0.029 %/0.069 % at 1 kHz against the 0.152 %/0.261 % already accepted at
  6 kHz. Raising it changed `dt` by 1.548x/1.095x at 1/2 kHz and `dx` not at
  all - so workspaces built before and after are NOT comparable at those tones.
- **`apertx > 0` is a source-centred TOTAL width.** Structure beyond `apertx/2`
  from a shot is not in that shot's model at all - size it from the range you
  want to resolve, not from the survey offsets, or you will measure the aperture
  and call it physics.
- **TE2D has no anisotropy input and correctly needs none** (Ey is horizontal,
  so VTI is degenerate with Sg/Ep). The old `A` key is REJECTED with a fatal
  error; write no anisotropy file.
- **Every tensor component must be calibrated on the SAME Earth model, AT EACH
  FREQUENCY.** `fdtd_analytic_calibration_by_source` is last-write-wins per
  source, so it is easy to end up with Kx calibrated on the lateral average and
  Kz on the homogeneous halfspace - different C and different sigma SCALES,
  silently. Step 02 now warns at the point of the mistake
  (`calibration_consistency_warning`) and `inversion_1d.tensor_calibration`
  still refuses it at inversion time. The comparison is deliberately WITHIN a
  frequency: per-frequency datasets legitimately differ in `rho_ohm_m`
  (26.09/27.60/29.04/29.85 at 1/2/4/6 kHz) because each grid resamples the same
  `sg.rss` differently.
- **`|C| ~ dx^2`, so raw `|C|` is meaningless across datasets with different
  grids.** Measured 2.61/1.95/0.90/0.64 at 1/2/4/6 kHz for `dx` =
  1.6/1.4/0.95/0.8 m. Compare `C/dx^2`; the real per-frequency spread is 2.04 %,
  not 300 %.
- **`mpiEminvTE2d` takes ONE `source_type` per run.** Joint multi-source FWI
  needs that to become per-shot upstream in rockem-suite.

## Measure before asserting

This project has repeatedly produced confident wrong conclusions from plausible
mechanisms. State a hypothesis, design the cheapest test that could falsify it,
and report what happened - including when it did not improve. Compare relative,
normalised quantities across codes with different source normalisations, and
compare COMPLEX values: an amplitude-only comparison is invariant under
conjugation and cannot tell a sign convention from a time convention.

The strongest available check, because this is a synthetic study: **does the
TRUE model fit the data?** It is what exposed a phase error of up to 21.6 deg
that every calibration-side check had missed (true-model chi-squared 381.8 ->
0.028). Use it whenever an inversion looks wrong.
