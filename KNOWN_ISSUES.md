# Known issues and unfinished work

Everything here is either broken, incomplete, or a limitation that will bite
someone. Each entry says what is wrong, what it breaks, and what fixing it
takes. Verified against the code at the time of writing, not asserted from
memory.

Ordered by how likely it is to cost someone a day.

---

## 1. Steps 04 and 06 do not know the acquisition matrix exists

**Status: broken for multi-dataset workspaces.**

Step 01 can now build one dataset per (frequency, source) pair, and Step 02 has a
dataset dropdown. Steps 04 (2D results) and 06 (1D results) have neither - they
read `CONFIG.fwd_2d_dir` directly and will find no `sg.rss`/`Data/` there when
the matrix layout is used, because the datasets live in subdirectories.

Verified: `grep -c "manifest\|iter_datasets"` returns 0 for both notebooks.

**Fix:** give each a dataset dropdown built from
`headless.iter_datasets(CONFIG.fwd_2d_dir)`, the same pattern Step 02 uses
(`_select_dataset` rebinding the path constants).

**Workaround until then:** use the historical single-dataset layout (one
broadband run, one source), which still writes straight into
`workspace/2D/forward/`.

---

## 2. `workshop_report.py` reports one dataset

**Status: incomplete.**

The LaTeX workflow report reads a single `setup_metadata.json` and knows nothing
about `manifest.json` (verified: 0 matches). On a matrix workspace it will
report whichever dataset it finds, or fail.

**Fix:** iterate datasets and emit a section per dataset, or take a
`--dataset` argument.

---

## 3. Joint multi-source 2D FWI is not possible with the current engine

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

## 4. The 2D multi-scale head-to-head has never been run

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

## 5. `inversion_tuning.split_objective` is still a two-component Kx misfit

**Status: stale relative to the tensor inversion.**

The DE-budget and Tikhonov-lambda tuners in Step 05 decompose the misfit
themselves (`obs_hx_gain`/`obs_hz_gain`, verified 2 matches) and call
`forward_analytic_for_tx`, which is Kx-only. With the full tensor selected, the
tuners therefore tune against a DIFFERENT objective from the one being
minimised, so their recommendations no longer correspond to the actual run.

**Fix:** route `split_objective` through `inversion_1d.tensor_objective` with
the same `components` as the run.

---

## 6. BlockInv cannot fit the full tensor

**Status: refuses explicitly (does not fail silently).**

`_invert_single_tx_blockinv` packs exactly the Kx pair into its data and
Jacobian vectors. Selecting Czx/Czz with that optimizer now raises
`NotImplementedError` rather than quietly fitting half the request. Use
Differential Evolution or Dual Annealing for the tensor.

**Fix:** generalise `_pack_complex` and the Jacobian assembly to N components.

---

## 7. The Step 05 QC overlay and the empymod fallback are Kx-only

**Status: incomplete.**

- `forward_analytic_true_model_for_tx` (the true-model QC curve) calls the Kx
  forward only, so with tensor components selected the overlay shows Kx curves
  against tensor data.
- `empymod_line_source.forward_empymod_line_gains` synthesises `ab=44`/`ab=64`,
  i.e. Kx only. The contrasted-interface fallback therefore cannot serve a Kz
  component. It is a rarely-hit fallback (~0.6 % of draws), but it will be wrong
  when hit.

---

## 8. `calibration_for_inversion` returns a single C array

**Status: works today, blocks per-frequency inversion.**

It returns one `C` array indexed by the metadata's `flist_hz`. With
per-frequency datasets each frequency has its OWN calibration on its own grid,
and they genuinely differ - measured **+2.04 % at 1 kHz** against the broadband
value, against 0.39 % scatter, with the same-grid control reproducing to 0.001 %
(so it is real grid dependence, not an assembly bug).

Anything that mixes per-frequency datasets must therefore carry per-frequency C
rather than averaging. `inversion_1d.tensor_calibration` already does this per
SOURCE; the per-FREQUENCY equivalent does not exist yet.

---

## 8b. Calibrations are stored per source with no Earth-model consistency check

**Status: guarded now, but the stored metadata WAS wrong.**

`save_calibration_to_metadata` keeps the last calibration per source under
`fdtd_analytic_calibration_by_source`, last-write-wins, with no record that the
sources must agree. Running the lateral-average method for Kx and the
homogeneous one for Kz therefore left a file that LOOKED complete but mixed a
28 Ohm-m reference with a 1 Ohm-m one - different C and, worse, sigma on a
different amplitude scale.

Found by the tensor test: the Czz residual came out at **8 sigma** while every
other component sat under 0.2, purely because its sigma came from the wrong
Earth. `inversion_1d.tensor_calibration` now refuses with an explicit message
naming both methods and their rho_ref.

**Remaining gap:** Step 02 still lets you produce that state - it does not warn
when you calibrate one source with a different method from another. The guard is
downstream of the mistake.

---

## 9. Interface snapping is off by default, and would only be a partial fix

**Status: quantified, not fixed.**

The inversion fits continuous interface depths against FDTD data whose
interfaces are quantised onto the FD grid. Measured: half a cell (0.4 m) moves
`|Hz|` by up to **4.95 %** and `|Hx|` by 2.34 % at 6 kHz, against a 3 %
uncertainty floor; the actual placement error is 0.34 m rms / 0.50 m max.

`unpack_model_params(snap_dz=...)` exists but defaults to off, because snapping
candidates to cell FACES only partly fixes it: `sg.rss` is resampled with LINEAR
interpolation, so an FD interface is a one-cell ramp whose midpoint sits between
faces. Making snapping exact also requires resampling the model blockily
(nearest, not linear).

---

## 10. The `eps_r` cap is left binding at low frequency

**Status: a measured, un-taken saving.**

`eps_r` clips at `eps_r_cap = 1000` for the 1 and 2 kHz runs (uncapped: 2397 and
1198), throwing away part of the low-frequency time-step gain. Removing the cap
is worth ~1.10x overall and costs 0.029 % / 0.069 % bias at 1 kHz - well inside
the 0.152 % / 0.261 % the 6 kHz run already accepts. Safe by the workshop's own
standard, simply not enabled.

---

## 11. The 3D / ADI path is unvalidated

**Status: parses, never checked.**

`mod3d.cfg` used to be rejected outright by current rockem-suite (the retired
`A` anisotropy key); that is fixed and it now parses. But no Green's-function
validation has ever been run against `mpiEmmodADI3d` from this workshop. Treat
2D TE as the only supported path.

---

## 12. Cxz-vs-Czx look-ahead distances are partly a threshold artifact

**Status: a caveat on a reported result, not a bug.**

`Czx` appears to detect the fault ~40 % further ahead than `Cxz` (112 m vs 76 m
at 1 kHz). They are the same physical quantity under reciprocity and differ only
through lateral structure. Most of the gap is that `Czx` is read on **Hx**
(calibration scatter 0.064 %) while `Cxz` is read on **Hz** (2.649 %) - a 41x
looser threshold, because Hz-from-Kx at zero depth offset IS the near-null.

Improving the Hz calibration (more receivers, or a geometry where Hz is not a
null) would make the comparison fair.

---

## 13. Older 1D inversion runs are not comparable with new ones

**Status: unavoidable consequence of two fixes.**

The extraction-window fix (`n_periods_extract` now excludes the source ramp) and
the time-base fix (absolute-time windows plus the exact sub-sample correction)
both change the observed channel gains. Any `workspace/1D/inversion/OneDRunN/`
produced before them was fitting phases wrong by up to 21.6 degrees, plus ~173
degrees of aliasing at 6 kHz.

The chi-squared values of 1.90-9.99 previously quoted as a benchmark belong to
that regime and should not be compared against current numbers.
