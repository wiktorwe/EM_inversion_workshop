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

## 1. The 2D multi-scale head-to-head has never been run

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

**Also fixed while wiring joint multi-source FWI in: the ladder could not
chain.** The engine writes accepted models to `Results/sg_up.rss-<iter>`
(`saveResults`, `inversionBase.h:28`), but `multiscale_2d_run.find_output_model`
globbed the run directory NON-RECURSIVELY, so it never looked in `Results/`. It
returned `None`, `prev` never advanced, and the ladder would have run as N
independent inversions from the uniform start model. It now goes through
`workshop_report.latest_sg_up_file`, the same reader the notebooks use.

Measured while checking this, because the obvious guess is wrong: the engine
CREATES `Results/` itself when it first saves a model - a run directory without
it produces `Results/sg_up.rss-1` exactly like one with it. `Local/` is likewise
not the problem: the template sets `incore = "true"`, so the checkpoints stay in
memory and `Snapfile` is never written. Neither directory has to be staged, and
the missing-model symptom was the reader alone.

---

## 2. BlockInv converges to a local minimum far from the global one

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

## 3. The 1D inversion's depth resolution is limited by the FD grid, not by noise

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
| true model, no snapping | 0.0239 |
| true model, snapped per frequency | 0.0244 |

**The true-model chi-squared does not decide this**, and it is recorded here so
nobody re-derives a conclusion from it: with `sigma` on the source's field
scale the two are the same number to within the spread across transmitters. The
evidence for snapping is the RECOVERED MODEL, below.

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

## 4. The survey reads the cross-couplings on their null

**Status: measured. The cure is an acquisition change, so it is left to the
user.**

`Cxz` and `Czx` vanish under ONE condition, and it is a SYMMETRY condition, not
a limit on the model class:

> the receiver sits at the source depth, AND `sigma(z)` is mirror-symmetric
> about that depth.

Reflecting z about the source plane flips the sign of Hz for a Kx source, so an
invariant medium and an in-plane receiver force the field to equal minus itself.
A homogeneous whole space is the trivial case of that symmetry, not a separate
rule. The shipped survey is colinear (`off_z = 0`), so it satisfies the first
half exactly and lands wherever the Earth's own asymmetry puts it.

Measured with `scripts/experiments/cross_coupling_geometry.py` at 4 kHz, near
receiver. Breaking the SYMMETRY, receiver still at the source depth:

| stack (source in the middle layer) | \|Cxz/Cxx\| |
|---|---|
| `100 / 30 (src) 30 / 100` - symmetric | 3.7e-13 (numerical zero) |
| `100 / 30 (src) 30 / 101` - 1 % resistivity asymmetry | 1.7e-05 |
| `100 / 30 (src) 30 / 120` - 20 % resistivity asymmetry | 2.9e-04 |
| `100 / 30 / 100`, layer centre 2 m off the source | 5.5e-03 |
| `100 / 30 / 100`, layer centre 5 m off the source | 1.5e-02 |

Breaking the GEOMETRY, moving the receiver off the source depth:

| depth offset | homogeneous | asymmetric layers |
|---|---|---|
| 0 m | **0** (exactly) | 3.29e-02 |
| 0.5 m | 8.08e-02 | 6.36e-02 |
| 2 m | 3.32e-01 | 3.23e-01 |
| **5 m** | **9.70e-01** | **1.01e+00** |
| 10 m | 4.50e+00 | 3.90e+00 |

So a 1D model produces these components perfectly well. Asymmetry of the Earth
about the transmitter lifts them off the null on its own; **five metres of depth
offset makes them the same size as the co-component, with no layering at all.**
The reason they carry little 1D information on this survey is that it sits on
the one geometry that suppresses them.

On the shipped production data the cross-couplings sit at 7e-4 to 1.3e-3 of
their co-components, which is where an almost-symmetric Earth read on the
symmetry plane puts them.

**This costs the inversion nothing, and must not.** `sigma` for every component
of a source is the relative error budget applied to that SOURCE'S FIELD SCALE
(`inversion_1d.amplitude_scale`), not to the component's own magnitude - a
half-cell interface move or the `(dx/r)^2` kernel error perturbs the field at
the receiver, and the field is the co-component. So a near-null datum is
compared against the error of the field it lives in, and contributes what it
should. Measured at Tx 0, 2 kHz, cost in chi-squared of being wrong by 100 % of
the component:

| component | \|obs\| | sigma | chi2 per datum |
|---|---|---|---|
| Cxx | 1.05e-01 | 1.34e-03 | 6.1e+03 |
| Czz | 1.20e-01 | 7.99e-03 | 2.2e+02 |
| Cxz | 9.11e-05 | 7.00e-03 | **1.7e-04** |
| Czx | 8.90e-05 | 1.53e-03 | **3.4e-03** |

The true model confirms it end to end: reduced chi-squared **0.0239** over 5 Tx
and all four components (`scripts/experiments/true_model_check.py`), of which
Cxx contributes 0.083 and the two cross-couplings 0.001 and 0.010. The fit is
driven by the components carrying signal, and the near-nulls neither help nor
hurt until they rise above the field's own error.

**What is left is an acquisition observation, not a defect.** Because these
components are suppressed here, they carry little 1D information, and what is
in them is mostly NOT 1D - which is exactly why `experiments/lookahead.py` uses
them to see the fault ahead of the bit. Giving the receivers a few metres of
depth offset would make them full-sized 1D observables, and would remove the
lopsided quantisation floor in section 7 at the same time. Nothing in the
workshop forbids it: `forward_1d_gains` takes per-receiver depths, and the
analytic and FDTD paths both handle it.

Notebook 05 can also weight each tensor component directly (`w Cxx` ... `w Czz`
-> the cfg's `w_Cxx` ... `w_Czz`, `inversion_1d.component_weights`) if you want
to exclude a component outright; `n_tensor_data` divides by the weighted count,
so the reduced chi-squared follows. That is a preference, not a correction -
the budget already gives every component its right weight.

---

## 5. The GUI manual describes an older version of the workshop

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

## 6. The 3D / ADI path is unvalidated

**Status: parses, never checked.**

`mod3d.cfg` parses, but no Green's-function validation has ever been run
against `mpiEmmodADI3d` from this workshop. Treat 2D TE as the only supported
path.

---

## 7. The reported Cxz-vs-Czx look-ahead distances predate the current threshold

**Status: the threshold is fixed in code; the numbers need re-running.**

`Czx` was reported to detect the fault ~40 % further ahead than `Cxz` (112 m vs
76 m at 1 kHz). They are the same physical quantity under reciprocity and differ
only through lateral structure, so a gap that large is a measurement artifact.
It came from the DETECTION THRESHOLD: `Czx` is read on Hx and `Cxz` on Hz, and
both thresholds were taken from the calibration's residual scatter - 0.064 % on
Hx against 2.649 % on Hz, a 41x difference produced by fitting Hz on its own
near-null.

`lookahead.py` and `fault_couplings.py` now threshold on
`fault_couplings.departure_thresholds` instead: the part of the analytic error
budget that does NOT cancel between the fault run and the reference run, which
is the interface quantisation. The two runs share a grid, a survey and a time
step, so the `(dx/r)^2` kernel error, the stencil factor and the solver's kx
quadrature are identical in both and divide out of a departure ratio.

**This makes the threshold defensible, but it does NOT close the gap at 1 kHz.**
The new floors, on the production grids:

| tone | Hx floor | Hz floor | Hz/Hx |
|---|---|---|---|
| 1 kHz | 0.116 % | 5.78 % | **50x** |
| 2 kHz | 0.528 % | 6.58 % | 12x |
| 4 kHz | 2.35 % | 7.70 % | 3.3x |
| 6 kHz | 4.02 % | 8.53 % | 2.1x |

The ratio collapses to 2-3x at the top of the band, but at 1 kHz - the tone the
112 m / 76 m comparison was made at - it is 50x, no better than the 41x it
replaces. The reason is no longer a fitted null but a physical one: Hz at zero
depth offset IS the near-null of section 4, so a half-cell interface move
changes it by a much larger FRACTION than it changes Hx. A derived floor
reports that honestly instead of hiding it, but it cannot remove it.

**So the Cxz-vs-Czx comparison is still not fair at low frequency**, and the fix
is the one section 4 names: give the receivers a depth offset. That removes the
null and the lopsided floor together. Comparing the two couplings at 4-6 kHz is
sound in the meantime.

**The published 112 m / 76 m numbers were measured with the old threshold and
should not be quoted.** Re-measuring them costs about 90 minutes of FDTD on this
machine (see the timing note in `lookahead.py`); nothing else blocks it.

---

## 8. The FITTED calibration path is sensitive to its reference Earth

**Status: measured. Affects the `calibration_source="fitted"` escape hatch only,
not the inversion.**

The inversion computes `C` and its error budget
(`inversion_1d.resolve_tensor_calibration` defaults to `"analytic"`), so no
calibration Earth enters it at all. On that path the workshop's strongest
self-check passes comfortably: reduced chi-squared **0.0239** over 5 Tx and all
four tensor components (`scripts/experiments/true_model_check.py`, which also
defaults to the analytic calibration).

The FITTED path is a different matter. Its `C` and `sigma` come from an FDTD run
on a reference Earth, and that choice moves the answer: `lateral_average_true`
gives 0.0296 where `homogeneous_rho_min` at 1 Ohm-m gives 3.92, against a
~30 Ohm-m production model. **So if you use the escape hatch for an A/B, use
`lateral_average_true`** - `scripts/run_matrix.py` already defaults to it.
`homogeneous_rho_min` is much cheaper but is simply the wrong reference here,
and the 3.92 is the only number anyone has measured for it.

---
