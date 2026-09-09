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

## 7. Step 05 keeps its own `SETUP_META`, unlike every other notebook

**Status: fixed for every current path, but structurally different.**

Steps 02, 03, 04 and 06 all carry `_select_dataset`, which rebinds
`FWD_2D_DIR`/`SETUP_META` onto a real dataset directory. Step 05 does not - it
consumes the WHOLE matrix at once, so there is no "selected" dataset to bind
to. It therefore resolves its metadata through `active_setup_meta()` and
rebinds `SETUP_META` once at startup.

That worked around a real failure (`FileNotFoundError:
.../workspace/2D/forward/setup_metadata.json` from the lambda tuner, because on
a matrix workspace the forward ROOT holds only subdirectories), but it leaves
two different mechanisms in the repo for the same job.

**Fix:** give `iter_datasets` a documented "representative dataset" accessor and
have all five notebooks use it, instead of one rebinding globals and another
resolving lazily.

---

## 8. Notebook 03's second code cell relies on an injected `display`

**Status: harmless in Voila, a trap for anything headless.**

`scripts/validate_notebooks.py` now executes EVERY code cell (it used to run
only the first, so notebook 04's 959-line GUI cell was never validated at all).
That immediately exposed notebook 03 using `display(...)` without importing it -
legal in a live kernel, which injects it, and a `NameError` anywhere else.

The validator now injects `display` the same way a kernel does, so this passes.
The underlying dependence on kernel-injected builtins is still there and will
bite any future headless driver.

**Fix:** import `display` explicitly in the notebooks that use it.
