# Known issues and unfinished work

Everything here is either broken, incomplete, or a limitation that will bite
someone. Each entry says what is wrong, what it breaks, and what fixing it
takes. Verified against the code at the time of writing, not asserted from
memory.

Ordered by how likely it is to cost someone a day.

**Keep this file current.** It is part of the chain, like the code and the
skill: an entry that has been fixed must be deleted in the same change that
fixed it, and a limitation discovered while fixing something else must be added
in that same change. A stale entry is worse than no entry, because it sends
someone to re-investigate a problem that no longer exists - or, worse, lets them
trust something that is still broken because its entry was quietly deleted
without the fix. See RULE 2 in `.claude/skills/em-inversion-workshop/SKILL.md`.

---

## 1. Joint multi-source 2D FWI is not possible with the current engine

**Status: engine limitation, upstream.**

`mpiEminvTE2d` parses `source_type` as a single `int` and stores it as one
scalar on `InversionEmTE2D`. The `switch` that applies it already sits INSIDE
the per-shot loop (`lib/inversion/inversionEmTE2D.cpp:362`) but reads that
global, so every shot in a run gets the same source type.

`multiscale_2d.build_ladder(source_fields=...)` therefore CASCADES over sources
(Kx stage, then Kz stage starting from its model) rather than inverting them
jointly. A joint inversion would sum the Kx and Kz gradients before stepping; a
cascade lets the last source have the final say.

**Fix:** make `sourcetype` per-shot upstream in rockem-suite (read it from the
survey/data file, or accept a comma list indexed by shot). Small and localised -
the loop structure is already right.

---

## 2. The 2D multi-scale head-to-head has never been run

**Status: not run, deliberately, with the cost measured.**

One FWI function evaluation on the CHEAPEST stage (1 kHz, 1.6 m grid, 30 shots,
`-np 6`) measures **6.4 minutes**. An L-BFGS iteration costs one gradient plus up
to `max_linesearch` more, so 20 iterations is ~4.3 h for that stage and ~6.4 h at
6 kHz; the full ladder plus the single-stage baseline it must be compared
against is **1.5-2 days**.

The staging is complete and its structural pieces are verified (log-resistivity
grid handoff, knot and regularisation schedules). Only the comparison is
missing. A truncated run would not answer a question about where each strategy
converges, so none is reported.

**Related:** the 1-iteration cost run in `workspace/2D/inversion/ladder_1iter/`
was killed part-way and its outputs are incomplete.

---

## 3. BlockInv converges to a local minimum far from the global one

**Status: measured, and it is not a budget problem.**

BlockInv now fits any subset of the 2x2 tensor (that limitation is fixed), but
on this problem it lands roughly **35x worse than Differential Evolution**:

| optimizer | components | chi2 scored on all four |
|---|---|---|
| BlockInv | Cxx+Cxz | 32.79 |
| BlockInv | all four | 27.05 |
| Differential Evolution | all four | **0.76** |

Raising `block_max_iter` does nothing: 15, 40 and 100 give **bit-identical**
results, because the run stops on `min_dphi_percent`, not on the iteration cap.
It is a local Gauss-Newton method started from a random draw, and it is
converging - to the wrong minimum.

Reproduce with `scripts/experiments/blockinv_tensor_test.py`.

**What it is still good for:** deterministic, linearised parameter
uncertainties (`uncertainty_from_result`), which DE does not give you. Use DE
for the model and BlockInv for error bars on a model you already trust.

**Fix:** start BlockInv from the DE solution rather than a random draw, or
multi-start it. Neither is done.

---

## 4. Interface snapping is off by default, and would only be a partial fix

**Status: quantified, not fixed.**

The inversion fits continuous interface depths against FDTD data whose
interfaces are quantised onto the FD grid. Measured: half a cell (0.4 m) moves
`|Hz|` by up to **4.95 %** and `|Hx|` by 2.34 % at 6 kHz, against a 3 %
uncertainty floor; the actual placement error is 0.34 m rms / 0.50 m max.

`unpack_model_params(snap_dz=...)` exists but defaults to off, because snapping
candidates to cell FACES only partly fixes it: `sg.rss` is resampled with LINEAR
interpolation, so an FD interface is a one-cell ramp whose midpoint sits between
faces. Making snapping exact also requires resampling the model blockily
(nearest, not linear) - which changes the forward model and so invalidates every
dataset and calibration on disk. That is why it is not in the quick-fix set.

---

## 5. The 3D / ADI path is unvalidated

**Status: parses, never checked.**

`mod3d.cfg` used to be rejected outright by current rockem-suite (the retired
`A` anisotropy key); that is fixed and it now parses. But no Green's-function
validation has ever been run against `mpiEmmodADI3d` from this workshop. Treat
2D TE as the only supported path.

---

## 6. Cxz-vs-Czx look-ahead distances are partly a threshold artifact

**Status: a caveat on a reported result, not a bug.**

`Czx` appears to detect the fault ~40 % further ahead than `Cxz` (112 m vs 76 m
at 1 kHz). They are the same physical quantity under reciprocity and differ only
through lateral structure. Most of the gap is that `Czx` is read on **Hx**
(calibration scatter 0.064 %) while `Cxz` is read on **Hz** (2.649 %) - a 41x
looser threshold, because Hz-from-Kx at zero depth offset IS the near-null.

Improving the Hz calibration (more receivers, or a geometry where Hz is not a
null) would make the comparison fair. That needs new FDTD runs on a new
geometry, which is why it is still here.

---

## 7. On a homogeneous-calibrated matrix the TRUE model does not fit

**Status: measured, cause not yet confirmed.**

The workshop's strongest self-check is "does the TRUE model fit the data?" - if
it does not, no inversion result from that workspace means anything.

On a per-frequency matrix calibrated with `homogeneous_rho_min`
(3 frequencies x 2 sources, rho_ref = 1 Ohm-m), the true model beneath tx0
scores **reduced chi-squared 3.92** over all four tensor components. On the
single-dataset reference workspace, calibrated with `lateral_average_true`
(rho_ref 29.85), the same check gives **0.0036**.

Fixing the per-frequency `eps_r` bug moved it from 4.15 to 3.92, so that was
real but not the main term. Three candidates, none yet tested:

- the calibration Earth. `homogeneous_rho_min` at 1 Ohm-m is far from a ~30
  Ohm-m production model; `lateral_average_true` exists precisely for this.
  Testing it needs one FDTD calibration run per dataset.
- `sigma` from a homogeneous calibration may be on the wrong amplitude scale
  for the production data, which inflates chi-squared without any physics
  being wrong.
- the layered approximation beneath that transmitter.

**Do this first on a matrix workspace:** calibrate with
`1D lateral average of true model`, then re-run the true-model check. Until
chi-squared is below 1, treat inversion results from that workspace as
uninterpretable.

---

## 8. Two different mechanisms bind dataset paths

**Status: both correct now, and enforced by a test - but still two mechanisms.**

Steps 02, 03, 04 and 06 rebind their path globals with `_select_dataset`, each
declaring its OWN hand-maintained list in a `global` statement. Step 05 has no
selected dataset - it consumes the whole matrix - so it binds `SETUP_META` and
`SG_TRUE_PATH` to a representative dataset at startup via `active_setup_meta()`.

That split is what let a broken path ship: `SETUP_META` stayed pointed at the
forward root, which on a matrix workspace holds no `setup_metadata.json`, and
the lambda tuner died on it in front of a user. A second instance
(`SG_TRUE_PATH`) was still present afterwards and was found only when
`scripts/dev/chainsweep.py` was written.

The hand-maintained `global` lists are the remaining hazard: adding a path
constant to one of those notebooks without adding it to its `_select_dataset`
silently reintroduces the bug. `chainsweep.py` now fails on exactly that, so it
cannot ship - but the design still invites it.

**Fix:** one mechanism. A `dataset_paths(run_dir)` helper in `headless` that
returns the whole set as a namespace, used by all five notebooks, so there is no
per-notebook list to forget.

---

## 9. Step 05 still does `float(eps_r)` on a per-frequency array

**Status: broken on every acquisition-matrix workspace. Confirmed in the GUI.**

`headless.matrix_setup` returns `eps_r` as one value per frequency (measured
1198.3 / 599.2 / 399.4 at 2/4/6 kHz on this survey). `get_eps_r_used()` in
notebook 05 returns that array, `current_cfg()["eps_r"]` stores it, and the
analytic forward (`forward_1d_gains` / `inversion_1d._eps_for_freqs`) already
accepts it. The inversion run itself can finish.

Then every consumer that still wraps it in `float()` dies with:

```
TypeError: only length-1 arrays can be converted to Python scalars
```

Hit in the GUI:

| Action | Where `float()` is | What the user sees |
|---|---|---|
| Convergence check | `check_kx_convergence` → `layers_from_rho_thk` (`analytic_1d_forward.py`) | `Convergence check failed: only length-1 arrays...` |
| True-model QC overlay | `forward_analytic_true_model_for_tx` in notebook 05 | overlay fails; `on_refresh_qc` reports the TypeError |
| Export | `build_1d_run_summary` (`run_report.py`: `"eps_r_used": float(cfg.get("eps_r", 7.0))`) | `Export failed:` plus traceback. No `REPORT.md`, no summary JSON |

Same class, not yet clicked in this session but will fire on a matrix:

- notebook 06 `on_generate_syn`: `eps_r = float(summary.get('eps_r_used', get_eps_r_used()))`. Even after a successful export of a list, this line still crashes. If export never writes `eps_r_used`, the fallback is already the array from `get_eps_r_used()`.
- notebook 06 also collapses receiver depth with `rx_depth_m = float(rx_z[tr_idx[0]])`, the same first-receiver bug notebook 05 already removed from its own forward path.

A single-dataset workspace (plain float) does not hit any of these. That is why
`validate_notebooks.py` / a one-dataset `workspace/` do not catch it.

**Same root, silent rather than a crash** (collapsing a per-dataset value to one
band-wide scalar, which RULE 3 forbids):

- `setup_n_periods_extract()` in notebooks 05 and 06 takes `m["n_periods_extract"][0]`.
- `get_f_min_hz` in notebook 05 reads the representative dataset only via
  `default_f_min_hz_from_meta`.
- `extract_features` then passes that collapsed `n_periods_extract` into
  `load_tensor_features`, which uses the override for every dataset instead of
  each dataset's own `setup_metadata.json`.
- `on_export` does `float(cfg.get("n_periods_extract"))` and `float(cfg.get("f_min_hz"))`
  on those already-collapsed scalars, so they survive export as one number.

The inversion objective is not in this list: it already threads `cfg["eps_r"]`
through `inversion_1d` without `float()`. The run can look successful and then
fail at QC / export, or export a report whose `n_periods_extract` / `f_min_hz`
are whoever's `[0]`.

**Fix:** stop wrapping per-frequency values in `float()`. `build_1d_run_summary`
should store `eps_r_used` as a list aligned with `freqs_hz` (a scalar only when
there is one value). `check_kx_convergence` should build layers per frequency's
`eps_r` (it already passes the array into `forward_1d_gains`; only the
`layers_from_rho_thk` / `lam_max` leg still assumes a scalar). Notebook 05's
true-model overlay and notebook 06's synthetics must pass the array through to
`forward_1d_gains`. Do not take `[0]` of `n_periods_extract` / `f_min` /
`eps_r` and apply it across the band.

---

## 10. Visualisation selectors still pretend one dataset holds the whole band

**Status: broken on every acquisition-matrix workspace. The widgets are the
pre-matrix layout; the data is not.**

Step 01 now writes one dataset per **(frequency, source)** pair, named
`f{Hz}_{hx|hz}` (`headless.dataset_name`). Each dataset's `flist_hz` is a
single tone. The plot GUIs in 02 / 04 / 06 still offer a **dataset** dropdown
*and* a **frequency** dropdown *and* a **receiver-component** (Hx/Hz) dropdown,
as if frequency were a slice inside a broadband gather. It is not. Changing
frequency independently of dataset cannot show another tone of the same run -
that tone lives in a different directory, modelled with a different source.

What the user actually sees:

| Notebook | Widget | What it is filled from | What you get on a matrix |
|---|---|---|---|
| 02 data plot | `view dataset` | `iter_datasets` names (`f2000Hz_hx`, …) | the (freq, source) pair. This is the real selector. |
| 02 data plot | `comp_freq` (`frequency`) | that dataset's `Hx['freqs']` = its one-element `flist_hz` | **one option, the first (only) tone** |
| 02 data plot | `component_select` | Hx / Hz | receiver only; source is already baked into the dataset |
| 02 calibration | `cal_freq_select` (`cal freq`) | `freqs_hz` of the **first** in-memory calibration | **one option**. Changing `view dataset` does not even rebuild this list, and `on_view_dataset_change` never calls `update_calibration_plot`. |
| 02 calibration | `cal_comp_select` (`cal comp`) | Hx / Hz | receiver of the currently viewed dataset's source. No way to pick Kz vs Kx except by hoping `view dataset` and the in-memory cal dict stay in sync (`k.startswith(view_dataset.value)` against keys `'name [HX]'`). |
| 04 data compare | `dataset_select` + `comp_freq` + `component_select` | same pattern as 02 | frequency dropdown is a one-item list of the selected run's tone |
| 06 data compare | `dataset_select` | `_select_dataset` rebinds path globals | on a matrix, `on_load_real` **ignores** this and assembles the whole band via `load_tensor_features`. The dropdown looks like it chooses what you see; it does not. |
| 06 data compare | `comp_freq` | assembled `feats['freqs']` | this one actually lists every tone - because 1D stacks them - while the dataset dropdown next to it is a leftover |
| 06 data compare | `component_select` | Hx / Hz | receiver only. Source (Kx vs Kz, i.e. Cxx/Cxz vs Czx/Czz) is not a view control at all. |
| 05 QC | `qc_freq_select` | stacked inversion freqs | this is the exception: 1D *is* joint across the band, so picking a tone of the QC overlay is meaningful. There is no source/receiver selector; all fitted tensor components are drawn as columns. |

The 02 calibration panel still has HTML that documents a **`cal source`**
dropdown. That control was removed (calibration is an action on ALL datasets).
The copy is now a lie, and the remaining `cal freq` / `cal comp` pair cannot
replace it because frequency is not independent of dataset.

This is not "missing a frequency in the list" as a fill-in bug. The axis the
widgets are choosing on **no longer exists**. A dataset *is* a frequency and a
source. A second dropdown that iterates `flist_hz` of the current dataset
will always be length 1. Cramming every dataset onto one axis to avoid a
dropdown is still wrong (RULE 2, visualisation half - that was tried in 02/04
and reverted). The current three-way split is the other failure mode of the
same rule: selectors that do not correspond to what is on disk.

**What the view control should be.** One list whose options are every
combination of

- frequency
- source type (Kx / Kz, i.e. `source_field` HX/HZ)
- receiver type (Hx / Hz)

labelled so a human can read it, e.g. `2000 Hz · Kx → Hx`. That is the 2×2
magnetic tensor at one tone, which is what the matrix actually acquired.
Keep the within-gather selectors (`tx`, `local rx`, `trace idx`, `plot`
metric, run number) - those still vary inside one combination.

Notebook 05's QC is the one place a lone frequency dropdown still makes sense
(joint inversion, one model, several tones). It should still be able to pick
the tensor component being drawn, rather than always drawing every column.

RULE 2's visualisation examples currently list "view dataset, frequency
dropdowns, component (Hx/Hz) dropdowns" as separate widgets. That list is the
old layout. Updating the skill belongs in the same change as the refactor
(RULE 5), not before it.

**Fix:** one view-combination dropdown (freq × source × receiver) shared by
the plot panels in 02, 04 and 06, populated from `iter_datasets` × {Hx, Hz},
not from a single dataset's `flist_hz`. Delete the now-redundant `frequency` /
`cal freq` dropdowns. Make `view dataset` in 06 either drive the 1D compare
(it currently does not) or go away in favour of that combination list. Rewrite
the 02 calibration HTML so it no longer describes `cal source`.

---

## 11. Notebook 03's second code cell relies on an injected `display`

**Status: harmless in Voila, a trap for anything headless.**

`scripts/validate_notebooks.py` now executes EVERY code cell (it used to run
only the first, so notebook 04's 959-line GUI cell was never validated at all).
That immediately exposed notebook 03 using `display(...)` without importing it -
legal in a live kernel, which injects it, and a `NameError` anywhere else.

The validator now injects `display` the same way a kernel does, so this passes.
The underlying dependence on kernel-injected builtins is still there and will
bite any future headless driver.

**Fix:** import `display` explicitly in the notebooks that use it.
