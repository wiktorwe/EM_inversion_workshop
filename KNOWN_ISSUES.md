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
without the fix. See RULE 5 in `.claude/skills/em-inversion-workshop/SKILL.md`.

**"It would invalidate the data on disk" is never a reason for an entry to stay
here.** `workspace/` is regenerable output, not source - overwrite it, clean it,
rebuild it. That excuse kept the interface-quantisation entry alive on its own,
and the fix took minutes once it was dropped. Say that models built before a
change are not comparable with ones built after, re-measure whatever numbers the
change invalidates, and move on. See "THE WORKSPACE IS DISPOSABLE" in the skill.

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

## 4. The 1D inversion's depth resolution is limited by the FD grid, not by noise

**Status: the two mechanical errors are fixed and measured; what is left is a
property of the discretisation.**

Two things were wrong and are now not:

- `sg.rss`/`ep.rss` were resampled from the SEG-Y model with LINEAR
  interpolation, so every interface was a one-cell RAMP - the FD medium at an
  interface was neither layer. They are now resampled with `nearest`
  (`headless.build_forward_inputs` and notebook 01's `on_apply_outputs`, which
  must stay in step). Measured with
  `scripts/experiments/interface_snapping.py`: transition cells per interface
  went 1 -> 0, and the placement error now sits inside the half-cell bound in
  every dataset (max 0.400 / 0.375 / 0.300 m at 2/4/6 kHz) where the linear
  model exceeded it (0.500 m against a 0.400 m bound).
- the true-model reader placed every interface HALF A CELL TOO DEEP. It treated
  `.rss` samples as cell tops (`z0 + (k+1)*dz`); they are nodes, so a material
  boundary is at `z0 + (k+1/2)*dz`. The error was exactly `dz/2` in every
  dataset - 0.4 / 0.475 / 0.7 m at 6/4/2 kHz - and half a cell moves |Hz| here
  by 4.9-6.6 % against a 3 % floor, so every true-model overlay carried it.
  `inversion_1d.blocky_layers_from_trace` is now the one reader, cross-checked
  against `interface_snapping.py`'s independent measurement of the same depths.

Candidate interfaces are now snapped onto **each frequency's own** FD grid
(`analytic_1d_forward.snap_interfaces_to_grid`, threaded through
`forward_1d_gains` and enabled by default in notebook 05). Per frequency is not
a refinement - each dataset resamples the same `sg.rss` on its own `dx`, so the
same true interface at 6020.5 m lands at 6020.1 / 6020.875 / 6020.8 m in the
2/4/6 kHz models. Measured with `scripts/experiments/true_model_check.py` over
5 Tx on the full 2x2 tensor:

| | reduced chi2 |
|---|---|
| true model, no snapping | 0.0660 |
| true model, snapped per frequency | **0.0296** |

**Snapping is required, and the error budget is not a substitute for it.**
Measured on the co-components (`scripts/experiments/sigma_budget_ab.py`, 3 Tx,
DE popsize 10 / maxiter 30, seed 42), as recovered-model error against the
truth - NOT chi-squared, which is not comparable across rows because sigma
differs:

| configuration | rms error in log10 rho | chi2 (data) |
|---|---|---|
| fitted sigma + snapping | 0.8417 | 3.32 |
| **analytic budget sigma + snapping** | **0.3569** | **1.58** |
| analytic budget sigma, NO snapping | 0.9662 | 3.76 |

Two things follow. The analytic error budget (`scripts/modules/fd_error_model.py`)
more than halves the model error against the fitted sigma. And snapping is what
makes it work - removing it is worse than either. They are ALTERNATIVES for the
quantisation term, not partners: when the candidate is snapped the error is gone
from the residual, so `analytic_tensor_calibration` sets the budget's
quantisation term to zero. Charging for it twice measured 0.9357, worse than
snapping alone.

**What remains.** The FD grid still quantises the Earth: the deepest interior
interface in the inversion's parameterisation is pinned at the depth-window
edge and is not fitted, so it is not snapped, and a stack whose interfaces
cannot be centred on the transmitter is rejected outright by the layered solver
(`layers_to_stack` centres on `tx_depth_m`, so the finite stack is symmetric
about it). Neither limits this survey - chi2 0.03 is far inside the noise floor
- but both are real constraints on how finely a 1D model can be described here.

---

## 5. A 1D inversion cannot fit the cross-couplings

**Status: measured. Unfixed by choice - the fix is a modelling decision.**

`Cxz` and `Czx` are near-nulls in a layered Earth: a 1D model predicts them
essentially zero. In the production data they are NOT zero - they are created by
the 2D fault structure, which is exactly why `experiments/lookahead.py` uses
them to see ahead of the bit. So a 1D inversion is being asked to fit data its
model class cannot produce.

A FITTED sigma hides this, because on those components it is **larger than the
datum itself**. Measured on the rebuilt matrix, transmitter 0, near receiver:

| component | \|obs\| | sigma, fitted | sigma, analytic budget | ratio |
|---|---|---|---|---|
| Cxx 2 kHz | 1.05e-01 | 1.93e-03 | 1.34e-03 | 1.4x |
| Czz 2 kHz | 1.20e-01 | 2.33e-03 | 7.99e-03 | 0.3x |
| **Cxz 2 kHz** | **9.11e-05** | **1.44e-04** | 6.09e-06 | **24x** |
| **Czx 2 kHz** | **8.90e-05** | **1.05e-04** | 1.14e-06 | **92x** |

On the co-components the two agree within a factor 0.3-1.4, so the budget is
sound. On the cross-couplings the fitted sigma EXCEEDS the signal - it says
"these are noise, ignore them", so those components are nominally fitted and
effectively are not. That is the near-null geometry of §8 doing real damage.

The analytic budget gives them their true weight, and the 1D fit to them is then
bad: reduced chi-squared **342** over all four components, against **1.79** under
the fitted sigma. The data is no worse; the model is inadequate and now says so.

**A claim elsewhere in the docs rests on the fitted sigma.** The skill and
notebook 05 record that fitting all four components improves both criteria
(chi2 0.7596 vs 0.8140). That comparison was made where the cross-couplings were
nearly weightless, so it measured little. Re-run it before quoting it.

**Fix - a modelling decision, not a bug, so it is left to the user.** Notebook
05 weights each tensor component separately (`w Cxx` ... `w Czz` -> the cfg's
`w_Cxx` ... `w_Czz`, `inversion_1d.component_weights`), so excluding the
cross-couplings is `w Cxz = w Czx = 0`, and the reduced chi-squared follows,
because `n_tensor_data` divides by the weighted count. What is NOT provided is
the other option - a model-inadequacy term added to their sigma. Inventing that
term without deriving it would rebuild exactly what the analytic budget
removed: a noise estimate that is really a fudge factor.

---

## 6. The GUI manual describes an older version of the workshop

**Status: stale, deliberately not updated yet.**

`doc/gui_manual.tex` and the `gui_manual.pdf` built from it still describe:

- two per-dataset **Calibrate** buttons, where there is one batch action with a
  method dropdown;
- `C(f)` being fitted in Step 02 and stored in `setup_metadata.json` for
  Steps 05/06, where it is computed as `dx*dz*s(order)`;
- a Kx-only analytic solver, where both Kx and Kz are used;
- 1D run folders named `OneDRunN`, where they are `Run{N}`;
- `Run modelling (background)` and the old `dataset` / `frequency` /
  `component` view dropdowns.

The PDF is the copy people read and it is not rebuilt by any script here, so a
half-updated `.tex` beside a stale `.pdf` would be worse than a manual that is
known-stale in one place. **It is left alone until the workshop reaches a stable
release**, then rewritten and rebuilt in one pass.

Until then the notebooks point at `README.md`, which is current.

---

## 7. The 3D / ADI path is unvalidated

**Status: parses, never checked.**

`mod3d.cfg` parses, but no Green's-function validation has ever been run
against `mpiEmmodADI3d` from this workshop. Treat 2D TE as the only supported
path.

---

## 8. Cxz-vs-Czx look-ahead distances are partly a threshold artifact

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

## 9. The true-model fit depends strongly on the calibration Earth

**Status: measured. The entry that used to be here - "the TRUE model does not
fit", reduced chi-squared 3.92 - was about a matrix calibrated with
`homogeneous_rho_min` at 1 Ohm-m, and it no longer describes any workspace this
repo builds.**

`scripts/run_matrix.py` defaults to `lateral_average_true`, and on a matrix
calibrated that way the workshop's strongest self-check passes comfortably:
reduced chi-squared **0.0296** over 5 Tx and all four tensor components
(`scripts/experiments/true_model_check.py`), against 3.92 for the
homogeneous-calibrated matrix. The first of the three candidates listed in the
old entry - the calibration Earth - was therefore the main term, and
`homogeneous_rho_min` at 1 Ohm-m against a ~30 Ohm-m production model is simply
the wrong reference.

**So: calibrate with `lateral_average_true`.** `homogeneous_rho_min` remains
available and is much cheaper, but a workspace calibrated with it has not been
re-measured since the resampling change, and the 3.92 above is the only number
anyone has for it. Run the true-model check before trusting an inversion from
one.

---
