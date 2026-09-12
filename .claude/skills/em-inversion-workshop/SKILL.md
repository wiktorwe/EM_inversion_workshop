---
name: em-inversion-workshop
description: Use when working on the EM_inversion_workshop repo - the Voila/Jupyter workshop for 2D electromagnetic forward modelling and inversion that drives rockem-suite's explicit TE2D engine (mpiEmmodTE2d/mpiEminvTE2d). Covers the six GUI notebooks, scripts/modules, the FD design chain, the FDTD-vs-analytic calibration C(f), the 1D layered inversion, the acquisition matrix (per-frequency x per-source), and the measured numerical findings. Triggers on "workshop", "01_fw_setup", "setup_metadata.json", "mod.cfg", "inv.cfg", "calibration", "C(f)", "1D inversion", "channel gain", "steady_state_gains", "per-frequency", "Kz source", "look-ahead", "Fault_1.sgy".
---

# EM inversion workshop

**The rules, in priority order. Read them before editing anything.**

1. **Do not leave a bug behind** - no dead references, no dead controls.
2. **Actions run on everything; visualisation selects freely.**
3. **Step 01 is the setup; every other notebook remembers it.**
4. **Change the whole chain, or do not change it.** A written
   consequence table in this conversation is the gate. No table, no edit.
5. **The docs are part of the chain.**


## THE WORKSPACE IS DISPOSABLE. NEVER LET IT BLOCK A FIX.

**`workspace/` holds no source. Overwrite it, delete it, rebuild it - whatever
the change needs. Datasets, calibrations, inversion runs and reports are all
regenerable output, and none of them is a reason to narrow, defer or water down
a fix.**

This is a standing instruction from the repo's owner, and it overrides any
instinct to preserve what is on disk. If a correct change invalidates every
dataset and every calibration in the workspace, make the change.

It is written here because the opposite reasoning had already reached
`KNOWN_ISSUES.md` and kept a real bug alive: the interface-quantisation entry
argued that fixing it "changes the forward model and so invalidates every
dataset and calibration on disk. That is why it is not in the quick-fix set."
That was the ONLY thing keeping it unfixed. The fix took minutes once the
excuse was dropped, and it measurably improved the recovered model.

So:

- Do not propose a worse fix to avoid a rebuild.
- Do not leave an entry in `KNOWN_ISSUES.md` because fixing it costs a rebuild.
  "It would invalidate the data on disk" is not a status; it is a cost, and a
  small one.
- Do not ask permission to overwrite `workspace/`. It is gitignored, so it is
  never in a commit and never in a diff.
- DO say plainly, in the change and in the commit message, that models built
  before it are not comparable with ones built after. That is the real
  obligation - not preservation, but honesty about comparability.

Rebuilding is cheap and scripted:

```bash
python scripts/rebuild_matrix_workspace.py                    # Step 01, seconds
python scripts/run_matrix.py --method lateral_average_true    # model + calibrate
```

Measured on this machine: 3 tones x 2 sources = 6 datasets, modelled AND
calibrated in **9.0 min** total.

Two things to remember rather than worry about:

- `scripts/dev/chainsweep.py` needs a dataset to build its fixture from, so
  with an empty workspace it reports **NOT VERIFIED**, not PASS. Rebuild, then
  re-run it. The other three sweeps pass on an empty workspace.
- The numbers quoted in `KNOWN_ISSUES.md` and `doc/numerics_findings.md` are
  measurements, not fixtures. If a change invalidates them, RE-MEASURE and
  update them (RULE 5) - do not preserve stale data to keep an old number true.


## RULE 1 - DO NOT LEAVE A BUG BEHIND

**Never hand over code with a known-broken reference, a dead symbol, an
unreachable branch, or a control that does nothing. Not once. Not "I'll fix it
next round".**

Every bug shipped from this repo has been the same shape: something was deleted
or renamed, and a reference to it was left behind. Deleting a widget and leaving
a handler that reads `.value` off it is a `NameError` in front of the user.
Deleting a button and leaving its `bind_button_with_feedback` call is a
`NameError` at import. These are not subtle - they are found by looking.

**After every edit, before saying anything is done, run all three:**

```bash
python scripts/validate_notebooks.py     # executes every code cell
python scripts/dev/bugsweep.py           # undefined names + unbound buttons
python scripts/dev/handlersweep.py       # CALLS every handler in every notebook
python scripts/dev/chainsweep.py         # every notebook against a MATRIX workspace
python scripts/dev/docsweep.py           # PROSE that describes code that is gone
```

**These sweeps are blind to the most common failure here.** They check names,
execution, handler exceptions and path binding. NONE of them can see a sentence
that has become false, or a guard that still demands a dependency the code has
dropped. `C(f)` became computed while notebook 02's panel still told the user it
was what Steps 05/06 read, the README said the same, and the guard that used to
be called `require_global_calibration` still RAISED - so Step 05 refused to
invert on an uncalibrated workspace. The other sweeps passed the whole time. So:

- `docsweep.py` closes the mechanical half: a backticked symbol in prose that no
  longer exists in code, a temporal word in GUI text, and a notebook section its
  own intro cell never mentions. It cannot judge whether a true-sounding
  sentence is still true.
- For the semantic half there is no tool, only a rule: **when you change what an
  input IS, delete that input and run the app.** Removing every calibration
  block from the workspace and re-running the inversion is what found the guard
  breakage described above; no sweep did.
- And re-read the plan before saying "done", accounting for every step. That
  breakage existed because a step that was written down ("rewrite the panel
  HTML") was simply never executed.

`validate_notebooks.py` alone is NOT enough: it executes the cell but never
clicks anything, so a `NameError` inside a callback passes it and still breaks
for the user. That is how the `cal_freq_select` bug shipped.

- `bugsweep.py` parses each notebook and reports names that are used but never
  defined, and Buttons that are never bound to a handler. It is the cheap check
  that catches "deleted a widget, left the reference".
- `handlersweep.py` executes every `on_*` / `update_*` / `refresh_*` function
  and fails on `NameError`/`AttributeError`. Other exceptions are expected
  (empty workspace) and ignored. It pins a plotly renderer and stubs
  `Figure.show`, because `fig.show()` OPENS A BROWSER TAB outside a notebook -
  do not remove that guard, it once spammed the user's browser.
- `chainsweep.py` builds a synthetic MATRIX workspace and executes every
  notebook against it, failing if any dataset artifact stays bound to the
  forward root. It walks the ATTRIBUTES of objects in the notebook namespace,
  not just bare `Path` globals, because the notebooks now hold one
  `headless.DatasetPaths` (`DS`) rather than a dozen loose constants - a
  bare-Path-only walk would report PASS on a notebook whose every artifact was
  root-bound. Its donor falls back to a dataset subdirectory, since a matrix
  root has no `setup_metadata.json` of its own. The repo's own `workspace/` is usually a single-dataset one,
  where the wrong path still exists - so the other three checks all pass while
  the notebook is broken for anyone with a real acquisition matrix. That is
  exactly how the `SETUP_META` bug reached a user.

Then, by hand: anything that spawns a thread needs its re-entrancy path tested
by clicking several times, and its progress panel checked for what it actually
displays. State that you ran these, with the output. "Should work" is not a
check.

Specific traps already paid for here:

- a widget deleted while a handler still reads it (`cal_freq_select`)
- a background worker that never sets the state its own status panel reads,
  so the panel says "not started" for the whole run
- a re-entrancy guard checked against state the worker never sets, so every
  click launches another batch
- `subprocess.run` where a handle was needed, so Stop had nothing to kill
- a plot reading a key (`fdtd_result`) that the on-disk JSON never carries,
  so it silently drew nothing
- `float(eps_r)` / `float(cfg.get("eps_r"))` after `eps_r` became a
  per-frequency array: the producer and the analytic forward were updated, the
  `float()` sites were not, and Step 05 died in the GUI (convergence, export,
  true-model QC) with `only length-1 arrays can be converted to Python scalars`

## RULE 2 - ACTIONS ARE ALWAYS ON EVERYTHING; VISUALISATION SELECTS FREELY

There are two kinds of control, and they have OPPOSITE rules. Get this
distinction right - both halves have been broken here, in both directions.

### Actions: never a choice. Always everything.

An action is anything that RUNS, INVERTS, CALIBRATES, MODELS, EXTRACTS or
WRITES. Step 01 is the only place frequencies and source components are chosen.
After it, **no action control may act on one frequency, one source or one
dataset.** They all act on the whole acquisition matrix.

Banned, no exceptions:

- a per-dataset "Run modelling" button - modelling runs every dataset
- "Calibrate (homogeneous)" / "Calibrate (lateral average)" per dataset, and any
  `cal source` dropdown - calibration does every dataset with its own source
- a `flist` / `f_min` / `n_periods_extract` input feeding an action - those come
  from each dataset's own `setup_metadata.json`, which Step 01 wrote
- a tensor-component selector on the inversion - fit every component present
- a "one run per frequency" / broadband switch - there is no broadband run and
  no broadband wavelet

Going from one broadband run to N frequencies x M sources must make the
workshop cheaper to USE, not turn it into N x M rounds of pick-and-click.

### Visualisation: selectors are REQUIRED. Do not remove them.

A plot control chooses what is DISPLAYED. It changes nothing on disk and runs
nothing. **Leave these alone, and add them where a figure would otherwise be
unreadable:**

- ONE `view` dropdown per plot panel, over the axes that actually exist on
  disk: **frequency x source (Kx/Kz) x receiver (Hx/Hz)**, labelled
  `2000 Hz . Kx -> Hx (Cxx)`. Built by `fd_visualization.view_combinations`
  from `iter_datasets` x {Hx, Hz}, and shared by Steps 02, 04 and 06.
- Tx index, local rx index, trace index, run number
- Step 05's `QC freq` and `QC comp`: the 1D inversion is JOINT across the band,
  so one model is fitted to every tone and picking a tone is meaningful there.
  It is the only place a lone frequency dropdown belongs.

The old layout - a `view dataset` dropdown AND a `frequency` dropdown AND a
`component` dropdown - was three controls on axes that are not independent. A
dataset IS a frequency and a source, so the `frequency` list was filled from a
one-element `flist_hz` and could never reach another tone; the SOURCE axis was
not selectable at all, so half the magnetic tensor was unreachable. Do not
re-split them.

Cramming every dataset and every frequency onto one axis to avoid a dropdown is
NOT what this rule asks for. It makes the figures useless. That mistake was made
once, in notebooks 02 and 04, and had to be reverted.

### The test to apply

Ask: **does this control change what gets computed or written to disk?**

- Yes -> it must not exist. The action does all of them.
- No, it only changes what is drawn -> keep it, and label it `view ...` so the
  next person does not delete it.

## RULE 3 - STEP 01 IS THE SETUP. EVERY NOTEBOOK REMEMBERS IT.

**The user enters the survey setup ONCE, in Step 01. No later notebook asks for
any part of it again - not the frequency list, not `n_periods`, not
`n_periods_extract`, not `f_min`/`f_max`, not `eps_r`, not the geometry.**

Step 01 writes all of it into each dataset's `setup_metadata.json`, and
`manifest.json` lists the datasets. Every later notebook READS those. A text box
in Step 02/03/04/05/06 that re-asks for a setup value is a bug even when it is
pre-filled from the metadata, because:

- the user can edit it to something the data was never modelled with, and
  nothing checks;
- with a per-frequency matrix there is no single right answer to pre-fill -
  each dataset has its OWN `flist_hz`, `f_min_hz` and `n_periods_extract`;
- it makes the setup look like a per-notebook choice when it is a property of
  the data on disk.

Read the value, show it read-only if it is worth showing, never offer it for
editing:

```python
meta = json.loads((Path(d['run_dir']) / 'setup_metadata.json').read_text())
freqs = np.asarray(meta['flist_hz'], dtype=float)
npx   = float(meta['n_periods_extract'])
f_min = float(meta['f_min_hz'])
```

Use the metadata key directly. Do NOT fall back to a widget, and do NOT use
`.get(key, default)` for a key Step 01 always writes - a silent default hides a
broken setup instead of failing where it can be seen. The key list Step 01
guarantees is in "`setup_metadata.json` IS the contract" below.

### And NEVER collapse a per-dataset value to one band-wide scalar

These keys DIFFER between datasets of a matrix, measured on the workshop survey:

| key | 2 kHz | 4 kHz | 6 kHz |
|---|---|---|---|
| `eps_r_used` | 1198.3 | 599.2 | 399.4 |
| `dx_model_target_m` | 1.40 | 0.95 | 0.80 |
| `dt_model_target_s` | 7.68e-8 | 3.69e-8 | 2.54e-8 |
| `f_min_hz` / `f_max_hz` / `flist_hz` | 2000 | 4000 | 6000 |
| `nt_model`, `wavelet_*`, `pml_heuristic` | all differ | | |
| the FD interface DEPTH `sg.rss` quantises to | 6020.1 | 6020.875 | 6020.8 |

Reading ONE of those and applying it across the band is a silent systematic
bias, not a rounding error. `eps_r_used` alone, taken from whichever dataset
happened to be representative, put **0.91 sigma** of bias on Hx at 6 kHz - most
of the assumed noise budget, on the best-resolved data in the survey - and
improved the true-model chi-squared from 4.1457 to 3.9171 when fixed. It looks
like a slightly poor fit, not a bug, which is exactly why it survived.

`headless.matrix_setup(fwd_root)` is the ONE resolver: it returns `freqs`,
`eps_r`, `f_min`, `n_periods_extract`, `dx` and `snap_origin_m` all aligned per
frequency, plus `by_source` / `meta_paths`. Steps 05 and 06 both use it. Do not
hand-roll another.

The analytic forward accepts `eps_r`, `snap_dz` and `snap_origin_m` as a scalar
OR one value per frequency, and `inversion_1d._slice_per_freq` slices all three
with `freq_mask` so a multi-scale stage gets the right one.

**Reports must not collapse them either.** `run_report.scalar_or_list` stores
`eps_r_used`, `f_min_hz` and `n_periods_extract` as a LIST aligned with
`freqs_hz` when there is more than one, and a scalar when there is one - so an
exported run says what it was actually fitted with.

## RULE 4 - CHANGE THE WHOLE CHAIN, OR DO NOT CHANGE IT

**When you change anything that something else depends on, you change every
consumer in the same edit, and you verify the chain end to end before you stop.**
A change that leaves any downstream step broken is not a partial success. It is
a regression, and it is worse than not having started, because the breakage
surfaces later and somewhere else.

Changing a dependency without its consumers is how this repo breaks, because the workshop *is* a
chain: Step 01 writes inputs -> Step 02 models and calibrates -> Steps 03/04
invert and display in 2D -> Steps 05/06 invert and display in 1D. Every step
consumes the previous one's files and metadata schema.

### Consequence analysis is the gate. No table, no edit.

Grep-after-the-fact is how this rule has been "followed" while still shipping
breakage. The `eps_r` type change is the type specimen: `matrix_setup` and
`get_eps_r_used` started returning a per-frequency **array**,
`forward_1d_gains` / `inversion_1d` were taught to accept it, and the work was
called done. The **name** was updated. The **contract** was not. Every site
that still did `float(eps_r)` died in the GUI (`check_kx_convergence` →
`layers_from_rho_thk`, `build_1d_run_summary`, notebook 05's true-model QC,
notebook 06's synthetics). It was KNOWN_ISSUES §9 until it was fixed; the
entry is gone, the lesson is not.

A type or shape change (scalar → array, one path → per-dataset, one `C` →
per-source) does not introduce a new symbol. Grepping the name and updating
the producer therefore looks complete and is not.

**Before the first edit**, write this in the conversation. Not in your head.
If it is not in the chat, you have not done the analysis, and you do not edit.

```
Change:       <what will be true that is not true now>
Old contract: <type, shape, who writes it, where it lives>
New contract: <type, shape, who writes it, where it lives>
```

Then one row per site that currently depends on the **old** contract:

| Site (file:function, or notebook handler) | Old assumption | If unchanged, breaks how |

Fill the rows with search, not recollection:

1. `grep -rn "<symbol>" --include="*.py" --include="*.ipynb" . | grep -v __pycache__`
   Notebooks are code: they will not show up in an import check.
2. **Also grep the operations that assume the old type**, or you will miss
   every `float()` wrapper. For a scalar becoming an array / list: `float(`,
   `int(`, `[0]`, `.item()`, `:.1f`, `:.4g`, `np.log10(`. For a removed
   argument: every caller, including fallbacks. For a removed widget: every
   `.value`. For a new capability: every branch, including error paths.
3. If the value is in the RULE 3 per-dataset table (`eps_r_used`, `f_min_hz`,
   `n_periods_extract`, `flist_hz`, `dx_model_target_m`, …), every
   `float(meta[key])` and every `m[key][0]` is a row.
4. A producer-only update (changed `matrix_setup`, left
   `run_report.build_1d_run_summary` doing `float(cfg.get("eps_r"))`) is the
   forbidden shape. It is not "I'll fix consumers next round".

Only after that table is in the chat do you edit. Every row is updated in the
**same** change, or the row says why the site is already compatible **and you
have read that code**. Then the RULE 1 sweeps, then the consumer a user
operates. An import that succeeds is not a behaviour that is correct.

If a numeric default changes, re-derive the threshold that depends on it. A
pass/fail threshold inherited from a different test is worse than none,
because it looks like a measurement.

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
| Made Step 05 matrix-aware | it kept `SETUP_META = <forward root>/setup_metadata.json`, a file that cannot exist on a matrix workspace - the lambda tuner died on it in front of the user | grep the notebook you edited for its OWN path constants, and run `chainsweep.py`, which fails on exactly this |
| Built one dataset per frequency, each with its own `eps_r_used` | Steps 05/06 kept reading ONE `eps_r` and applying it to the whole band - 0.91 sigma of bias at 6 kHz, visible only as a slightly worse fit | list which metadata keys VARY across datasets, then grep every reader of each one |
| Then made that `eps_r` an **array** in `get_eps_r_used` | every leftover `float(eps_r)` (`layers_from_rho_thk`, `build_1d_run_summary`, notebook 05 QC, notebook 06 synthetics) - TypeError in the GUI | type/shape change: grep `float(` / `[0]` / format of the OLD scalar, not just the name |
| Fixed those `float()` sites so Step 06's synthetics ran at all | the NEXT statement in the same handler, `int(real['Hx'].get('nt', 0))`, died on a matrix: `.get(k, default)` returns None when the key EXISTS and holds None, which an assembled result's `nt`/`dt` do | when you unblock a handler, RUN it to the end - the crash you fixed was hiding the next one |
| Made the 2D inversion a per-frequency sequence | notebook 03 staged a uniform `sg0` for every frequency and the run worker copied it unchanged, so every scale restarted from the same initial model | the experiment driver already had the handoff (`resample_model_log_rho`); grep the notebook's run loop for `sg0`, not only the module that implements the resample |
| Baked inv.cfg knobs only at Generate Inputs | Max iter looked dummy: Run copied the staged file, so every scale did template 20 | grep whether Run writes `inv.cfg`, not only whether Generate reads the widget |
| Replaced the per-notebook path globals with one `DS` | `chainsweep.py` only walked bare `Path` globals, so it would have reported PASS on a notebook whose every artifact was root-bound | when you change the SHAPE of what a check inspects, change the check in the same edit - a check that cannot see its subject reads as evidence |
| Made Step 05's features load on a matrix | `handlersweep.py` then actually reached `on_tune_de_budget`, a real 174 s DE run, and the sweep timed out | a sweep that calls every handler must BOUND the ones that optimise - `handlersweep` now patches the tuners to a single tiny budget rather than skipping them, because the lambda tuner is exactly the handler that shipped a bug |
| Added per-frequency interface snapping | the empymod fallback branch still forwarded the UNSNAPPED thickness, silently modelling a different Earth than the main path | follow the new argument into every branch, including error paths - this is the second time that exact branch has been missed |
| Made each 2D `Run{N}` one frequency | Step 04 listed Run0/Run1/Run2 with no tone or ladder, and the report compared the latest Run to the first forward dataset | a ladder is one attempt: `find_next_run_dir` was inside the frequency loop and `dataset.txt` lived at the Run root |
| Made each 2D `Run{N}` a multi-scale ladder (`ladder.json` + scale subdirs) | Step 04 still listed folder names, progress parsers looked at the Run root, the report took `sg_up` at that root, tensorsweep looked for `inv.cfg` there | `iter_stages` is the one reader; no `inv.cfg`-at-root fallback; old one-frequency trees were deleted rather than dual-format |
| Made the workshop report one document for the whole ladder | `--all-datasets` was writing `report/<dataset>/` and the default was the first forward dataset | one `make_workshop_report.py` writes `workspace/report/`; `--all-datasets`, `--dataset` and `match_2d_to_dataset` were deleted |
| Decoupled Step 04 scale from the data view | the view was source x receiver at the scale frequency, so Generate synthetics modelled one tone and a 2 kHz iterate could not be compared to a 6 kHz gather | Scale is the inverted model; view is frequency x source x receiver; Generate synthetics resamples the iterate onto every frequency's grid |

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

## RULE 5 - THE DOCS ARE PART OF THE CHAIN

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

### GUI TEXT IS TIMELESS. It says what a control DOES, and nothing else.

**Every string a user sees - markdown cells, `ipw.HTML`, `push_message`, widget
`description=`, status text - describes the PURPOSE of the control and HOW TO
USE IT. That is the whole brief.**

Banned in user-facing text, without exception:

- **Temporal words**: "now", "no longer", "used to", "previously", "formerly",
  "instead of", "this replaces". The GUI is not a changelog. A user reading it
  has no idea what it used to say and does not care.
- **Change notes and rationale**: why something was replaced, what it was
  before, what bug it fixed.
- **Measured numbers** ("worst deviation 0.430 %", "chi2 0.0296"). Those are
  measurements; they belong where measurements are recorded.
- **Cross-references to `KNOWN_ISSUES.md` entries** by number.
- **Limitations and current state.**

Where each of those goes instead:

| Content | Home |
|---|---|
| limitations, current state, "this is broken/partial" | `KNOWN_ISSUES.md` |
| parameter meaning and impact, longer descriptions | `doc/gui_manual.tex` |
| workflow, install, layout | `README.md` |
| measured numbers and how they were obtained | `doc/numerics_findings.md` |
| why the code must be the way it is - the CONSTRAINT and its reason | CODE COMMENTS, present tense (see below) |
| what the code used to be | THE COMMIT MESSAGE, and RULE 4's breakage table |

### AND DO NOT TRACK CHANGES IN DOCSTRINGS OR COMMENTS EITHER

**Git already records what changed. A docstring that says "this used to be X"
is a second history that nobody updates, and it MISLEADS THE NEXT READER -
including the next agent, which reads docstrings as statements of fact.**

That is not hypothetical. `get_global_calibration`'s docstring described a role
the function had already lost, and was read as current. A docstring saying
"this used to be `require_global_calibration`" also broke `docsweep` itself,
because a symbol mentioned only in its own history looked defined.

The line to draw:

- **BANNED** - the past: "this used to be X", "was changed from", "previously
  did Y", "the old version", "before the refactor".
- **KEPT, and required** - the present-tense CONSTRAINT and its reason:
  "windows on ABSOLUTE time, because production and calibration runs record at
  different `dtrec`". That is not history; it is why the code must stay this
  way, and deleting it invites the regression back.

Test to apply: strip every clause about the past. If the comment still tells you
what the code must do and why, it was a constraint - keep it. If nothing is
left, it was a changelog entry - delete it, and let the commit message carry it.

Where the history goes instead: the commit message, and - when it is a lesson
that must not be repeated - RULE 4's breakage table in this file. One place, not
scattered through the source.

`scripts/dev/docsweep.py` checks user-facing strings for temporal words. Comments
and docstrings are judged by the test above, not by regex, because a constraint
and a changelog entry can use the same words.

### A NOTEBOOK'S INTRO CELL IS A DELIVERABLE, AND IT GOES STALE SILENTLY

**When a notebook gains, loses or repurposes a SECTION, its intro markdown cell
is a mandatory site in the consequence table. Every section the notebook has
must be visible in the sentence that says what the step is for.**

This is the one failure mode no symbol-level sweep can reach on its own. An
intro that describes two of five sections contains no dead symbol and no
temporal word: every word in it is TRUE. It is still wrong, and it is the first
thing a user reads - notebook 02 described modelling and plotting while also
running the whole acquisition matrix and extracting the channel gains Steps
05/06 invert.

`docsweep.py`'s third check closes the mechanical half: every numbered section
heading must have one of its distinctive words in the intro, stem-matched. It
cannot judge whether the sentence describes the section WELL - that is the
consequence table's job.


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
  - **there is no broadband run and no single-dataset action, anywhere.**
    `build_forward_matrix` always writes one dataset per frequency per source;
    the `split_by_frequency` switch and the `One run per frequency` checkbox are
    gone. A broadband grid is set by the highest tone and its record length by
    the lowest, so it pays for both ends at once - measured 621.9 s against
    312.1 s split. Step 01 also builds one single-tone wavelet PER FREQUENCY;
    there is no broadband wavelet to build or to display.
  - `iter_datasets` still falls back to treating the forward directory itself as
    one dataset. That is a READER for legacy workspaces already on disk, not a
    way to create one.

### Step 02 - forward modelling + calibration
- **imports** `fd_visualization`, `fdtd_analytic_calibration`, `headless`,
  `rockem_bridge`, `setup_defaults`, `workshop_config`
- **reads** a dataset's `mod.cfg` + inputs, `setup_metadata.json`, `manifest.json`
- **writes** `Data/{Hx,Hz}shot.rss`, `Data/processed/amp_phase_results.npz`,
  `calibration_{homogeneous,lateral_average}[_hz]/`, and the calibration blocks
  `fdtd_analytic_calibration` (active, Kx only) and
  `fdtd_analytic_calibration_by_source` back into `setup_metadata.json`
- **key chain facts**
  - **every action is on ALL datasets. There are no single-dataset buttons.**
    Modelling, gain extraction, calibration and saving each walk
    `headless.iter_datasets`; `headless.run_calibration_matrix` is the shared
    calibration loop and `scripts/run_matrix.py` does the lot headlessly. The
    per-dataset "Run modelling", "Calibrate (homogeneous)" and
    "Calibrate (lateral average)" buttons and the `cal source` dropdown were
    REMOVED - they made the operator carry the acquisition matrix in their head
    and could leave a workspace half-processed or mixed across Earth models.
  - the batch calibrates each dataset with the source it was MODELLED with, and
    uses ONE method for all of them.
  - the `dataset` dropdown that remains is a VIEW selector for the plots. It
    changes what you look at, never what runs.
  - `compute_gains_for_fd_outputs` -> `steady_state_gains` is THE extraction used
    by Steps 02/04/05/06 and the report. Its windowing rules (absolute-time
    alignment, ramp exclusion) are load-bearing for all of them.
  - calibration runs write `dtrec = dt_model`; production runs write
    `dtrec = 1e-5`. That asymmetry is why the extraction must align on absolute
    time.
  - only an HX calibration may take the ACTIVE slot - Steps 05/06 model a Kx
    line source and would otherwise be handed the wrong C.

### Step 03 - 2D inversion staging and run
- **imports** `fd_visualization`, `headless`, `inversion`, `multiscale_2d`, `workshop_config`
- **key chain fact** it stages and runs one JOINT inversion per FREQUENCY, over
  all of that frequency's source components, sequentially, in one action.
  `mpiEminvTE2d` takes a comma-separated `source_type`, so its work list spans
  (shot x source type) and the Kx and Kz gradients are summed before the step -
  one model update sees all four tensor components. The sequence is a ladder:
  lowest frequency first, and each frequency after the first starts from the
  previous inverted model (`Results/sg_up.rss-<iter>`), resampled onto this
  frequency's own grid by `multiscale_2d.handoff_starting_model`. Staging writes
  a uniform `sg0.rss` into every `input/<freq>_hx_hz/`; that is the
  first-frequency start. One click of Run inversion allocates one `Run{N}/`
  and puts each frequency in a subdirectory that is the engine cwd; `ladder.json`
  at the Run root lists the stages. The run worker overwrites
  `Run{N}/<freq>/sg0.rss` before each later frequency starts - copying the
  staged uniform `sg0` unchanged makes N independent inversions that only look
  like a ladder. A failed or model-less stage stops the batch, because later
  frequencies have nothing to start from. Inputs go to
  `inversion/input/<freq>_hx_hz/`, each scale records its group in
  `Run{N}/<freq>/dataset.txt`, and Stop halts the whole batch rather than
  letting the next frequency start. `headless.group_datasets_by_frequency` is
  the one grouping, shared with `multiscale_2d.build_ladder`.
- **key chain fact** Max iter, geps, apertx, dtx, dtz and Tikhonov are written
  into `Run{N}/<freq>/inv.cfg` at launch by `inversion.apply_inversion_controls`, from
  the live widgets. Staging copies `input/<freq>/inv.cfg`; leaving that copy
  unchanged is what makes Max iter look dummy - the engine reads the scale
  directory, not the widget. There is no per-frequency iteration picker: one
  cap for every scale. `geps` (gradient L2-norm, the GNORM column) is the
  scale-local stop; `0` disables it because gnorm is always > 0. The engine
  also stops on `xeps` / `fmin` / `max_linesearch` when those tests fire.
- **reads** a forward dataset: `mod.cfg` (for `order`, `lpml`, `pml_*`,
  `source_type`), `sg.rss`, `ep.rss`, `wav2d.rss`, `Data/{Hx,Hz}shot.rss`
- **writes** `workspace/2D/inversion/input/` and `Run{N}/`: `inv.cfg`,
  `sg0.rss`, `ep.rss`, `wav2d.rss`, `weight.rss`, `Sg_min/max.rss`, `Local/`,
  `Results/`, and the observed records - `Hx_data.rss`/`Hz_data.rss` for ONE
  source, `Hx_Hx_data.rss` ... `Hz_Hz_data.rss` (one per source x receiver) for
  a joint run. `inversion.find_observed_records` is the one reader for both
  forms; do not hardcode either. The Run root also carries `ladder.json`.
- **key chain fact** `prepare_inversion_inputs` PINS `order`, `lpml` and `pml_*`
  from the forward `mod.cfg`, and every forward run in a joint group must agree
  on them (they share one `inv.cfg`, so one grid, stencil and PML). Anything the
  forward run chooses that the inversion must match belongs in that list - a
  mismatch is a silently wrong gradient, not an error. `source_type` is no
  longer copied, because a joint run has several; instead each forward directory
  is ASSERTED to name the source it is filed under, so a mis-filed dataset fails
  loudly rather than fitting Kz data with a Kx source.

### Step 04 - 2D results
- **imports** `fd_visualization`, `headless`, `inversion`, `multiscale_2d`, `rockem_bridge`, `segy`, `setup_defaults`
- **reads** `Run{N}/` models + the forward data + `setup_metadata.json`
  (frequencies, `n_periods_extract`, SEG-Y template geometry)
- **writes** `workspace/2D/results/Run{N}/` SEG-Y exports, and QC synthetics into `Run{N}/pred/<scale>_iterNNN/<dataset>/` (does not overwrite engine `data_mod_*` in other stages)
- **key chain fact** it has TWO code cells - setup and GUI. Shared chrome is
  the ladder (`Run:`) and the scale (which inverted model: frequency of
  inversion plus iterate). Tabs split Models from Data so the view selector
  sits next to the gather plot. Scale selects the inverted model; the true
  model on the Models tab follows that scale's grid. The view dropdown is
  every frequency x source x receiver of the acquisition matrix, independent
  of the scale. Observed and synthetic gathers on the Data tab follow the
  view, which may be a different frequency. Generate synthetics from a
  selected iterate predicts ALL frequencies: the earth model is not a
  single-tone gather, so the iterate is resampled onto each frequency's
  grid. Export writes `workspace/2D/results/Run{N}/<stage>/`.

### Step 05 - 1D layered inversion
- **imports** `analytic_1d_forward`, `fd_visualization`, `fdtd_analytic_calibration`,
  `headless`, `inversion_1d`, `inversion_hybrid`, `run_report`, `segy`, `setup_defaults`
- **optimizer** is fixed to `de_blockinv_hybrid` (no dropdown): one DE run per Tx
  yields a 10–90% population envelope; the DE best is polished with BlockInv for
  the point model. Solver budget (popsize, maxiter, λ, BlockInv polish, seed,
  component weights, `n_jobs`) comes from `inversion_hybrid.HYBRID_DEFAULTS` via
  `apply_hybrid_defaults()` — the GUI exposes **model parameterisation only**
  (layers, depth range, ρ/thk bounds, background ρ, section grid).
- **reads** every dataset of the matrix, grouped by source: each per-frequency
  dataset contributes the ONE tone it was designed for, and
  `inversion_1d.load_tensor_features` stacks them in frequency order. A
  single-dataset workspace takes the historical path unchanged.
- **calibration** is assembled per frequency too
  (`calibration_for_inversion_multi`, and `resolve_tensor_calibration` accepts a
  per-source mapping of metadata lists). Reusing one dataset's C across a band
  it was not fitted on is a real error, not a rounding one.
- **every available tensor component is fitted.** If Step 01 built a Kz
  dataset, the full 2x2 tensor is what gets inverted. Fitting a subset of what
  was acquired throws data away, and it costs nothing to include a component
  that carries little: `sigma` is the error budget on the SOURCE'S FIELD SCALE
  (`inversion_1d.amplitude_scale`), so a near-null datum contributes in
  proportion to how far it stands above the field's own error - which for the
  shipped colinear survey is almost not at all.
- **writes** `workspace/1D/inversion/Run{N}/`: `REPORT.md`,
  `analytic_1d_inversion_summary.json`, `run_metadata.json`, NPZ with
  `envelope_rho_p10/p50/p90_tx{k}` grids and `section_rho_p10/p90` when built
- **key chain facts**
  - the forward is `analytic_1d_forward.forward_1d_gains(source_field=...)`;
    it MUST match the source that produced the data.
  - `inversion_1d` holds the one true parameterisation and misfit. Notebook 05
    imports it - do not re-add a local copy. Do not re-expose DE/lambda tuners
    or solver knobs in the GUI.

### Step 06 - 1D results
- **imports** `analytic_1d_forward`, `fd_visualization`, `fdtd_analytic_calibration`,
  `headless`, `run_report`, `segy`, `setup_defaults`
- **reads** `Run{N}/` summaries (including `run_metadata.json`'s
  `data_convention`) + the forward data
- **writes** `workspace/1D/results/` SEG-Y exports
- **key chain facts**
  - it carries the same `view` dropdown as Steps 02 and 04, and its SOURCE
    half is load-bearing: `on_load_real` assembles the whole tensor with
    `load_tensor_features` and the selector picks which source's half is drawn,
    with `get_calibration_C` and the synthetics following the same source.
    Before that it was hardcoded to Kx and `gains['HZ']` was computed and
    discarded, so Czx/Czz could not be displayed at all.
  - it WARNS when a run's `data_convention` is older than
    `run_report.DATA_CONVENTION`. Bump that constant whenever a change makes
    new channel gains incomparable with old ones, and say why in its comment.

### Out of band
`scripts/make_workshop_report.py` -> `workshop_report.py` reads the whole
workspace and writes **one** `workspace/report/workflow_report.tex` + figures.
It covers the acquisition matrix (every frequency and source), the selected
2D ladder (`--2d-run RunN` means `Run{N}/ladder.json` and every scale), and
the selected 1D run. There is not a report per frequency, source, or dataset.
2D observed-versus-synthetic figures overplot every ladder frequency on one
axes pair per source (`inv2d_data_{src}.pdf`); model/slice figures stay per
scale. `--list-datasets` lists the matrix and exits.

### THE PATH-BINDING CONTRACT (this is a link, and it was missing here)

On an acquisition-matrix workspace the forward ROOT contains **only**
`manifest.json` and one subdirectory per dataset. Every one of these belongs to
a DATASET, never to the root:

```
setup_metadata.json  sg.rss  ep.rss  wav2d.rss  Survey.rss  mod.cfg
runmod.sh  mpiqueue.log  Data/Hxshot.rss  Data/Hzshot.rss
```

**So no notebook global may be bound as `CONFIG.fwd_2d_dir / <one of those>`.**
It must be bound through a dataset, and there is ONE mechanism for that:

    headless.dataset_paths(run_dir) -> DatasetPaths

a frozen dataclass carrying `dir setup_meta sg ep wav2d survey mod_cfg runmod
clean_sh mpiqueue_log data_dir hx hz processed_dir amp_phase_npz`. Every
notebook holds a single global `DS` and reads `DS.hx`, `DS.setup_meta`, ...
`headless.select_dataset(root, name)` returns `(entry, DatasetPaths)` and is
what each notebook's `_select_dataset` calls.

Steps 02, 03, 04 and 06 rebind `DS` from the `view` selector. Step 05 consumes
the WHOLE matrix, so it has no selected dataset: it binds `DS` once to the
representative dataset via `active_setup_meta()`, and reads every dataset
through `matrix_dataset_dirs()` / `matrix_metadata_paths()`.

This used to be TWO mechanisms and eleven loose path constants per notebook,
rebound through a hand-maintained `global` list. Forgetting to extend that list
is how a root-bound path shipped three times: `SETUP_META` (found by a user),
`SG_TRUE_PATH` (found by `chainsweep.py`) and `FDMODEL_DATA_DIR` in Step 05,
which was never rebound at all and survived only because the matrix branch of
`extract_features` returned before reaching it.

This contract is the link that was missing from this map, and its absence is
why a broken one shipped: the map recorded what each step READS, but never how
the path gets RESOLVED, so "follow the chain" could not catch it. A map of
artifacts is not a map of the chain - record the mechanism too.

Enforced by `scripts/dev/chainsweep.py`. Run it after touching any path
constant, and do not rely on remembering this.

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

`analytic_1d_forward` <- `inversion_1d` <- `inversion_hybrid` <- `inversion_blockinv` <- {notebook 05,
`inversion_tuning`, `scripts/experiments/{multiscale_1d,tensor_1d_test,
blockinv_tensor_test,de_blockinv_hybrid}.py`}
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
- **`mpiEminvTE2d` takes a LIST of source types, and the 2D FWI is joint.**
  `source_type = "3,5"` spans (shot x source type) in one work list; the joint
  gradient IS the sum of the per-source gradients (measured 4.4e-08 relative L2
  by `scripts/experiments/joint_source_gradient.py` on its tiny model, and
  1.01e-07 once at production scale - 2 kHz, 30 shots, order 6), and it costs
  about the two single-source runs combined, not 4x. Three things follow.
  - The observed data is keyed `Recordfile_<SRC>_<REC>` - `Recordfile_HX_HZ` is
    Hz recorded from a Kx source. With ONE source `Recordfile_<REC>` remains the
    fallback and a pre-change `inv.cfg` runs byte-identically; that is a verified
    upstream regression, so do NOT modernise single-source configs to the pair
    form. `update_cfg_values` appends the pair keys via `allow_new` rather than
    the template carrying them.
  - Every record file must share trace count, trace order, per-trace coordinates
    and time axis. `apertx > 0` is a source-centred TOTAL width from each
    gather's own coordinates and ONE keymap is built from the first file, so a
    disagreement would give the same shot a different local model per source.
    The engine checks it at startup and names the file and trace index.
  - `misfit.rss` and `source_grad.rss` hold `ngathers * nsources` entries,
    SOURCE-MAJOR; `data_mod_*`/`data_res_*` gain a source tag ONLY when more than
    one source is listed. `inversion.available_synthetic_pairs` is the one reader
    for every naming form.
- **A joint run's four components are told apart by the SOURCE axis, and only
  `scripts/dev/tensorsweep.py` checks it.** Cxx/Cxz/Czx/Czz live in one run
  directory as `Recordfile_<SRC>_<REC>` inputs and `data_mod_<SRC>_<REC>`
  outputs, so every reader needs the source: `inversion.find_observed_records`
  and `inversion.available_synthetic_pairs` both take `source_field`, and Step
  04 passes its view's. Reading the Kx pair for a Czx view is not a NameError, a
  bad path or a false sentence, so the other four sweeps PASS while the panel
  mislabels half the tensor. It shipped that way once.
- **Scale is the inverted model; view is the gather.** Each scale inverts
  ONE tone; `Run{N}/<freq>/dataset.txt` records which, and `ladder.json` lists
  every scale of that attempt. Step 03 reads it for the true model it plots
  (per-frequency grids differ, and the colour limits come from it). Step 04's
  Models tab follows the SCALE's grid for the true model; observed and
  synthetic gathers on the Data tab follow the VIEW, which may be a different
  frequency. Generate synthetics resamples the selected iterate onto each
  frequency's grid, because the earth model is not a single-tone gather.
- **Step 03 is a frequency ladder in one `Run{N}`.** Each frequency after the
  first starts from `Results/sg_up.rss-<iter>` of the previous **scale
  subdirectory**, resampled onto this frequency's grid by
  `handoff_starting_model`. Staging writes a uniform `sg0` into every input
  directory because that inverted model does not exist yet; the run worker
  overwrites `Run{N}/<freq>/sg0.rss` before the later frequencies start. Copying
  the staged uniform `sg0` unchanged is the bug that makes every scale restart
  from the same initial model. Allocating a new `Run{N}` per frequency is the
  bug that makes Step 04 unable to follow the ladder.
- **Step 03 inversion knobs are applied at launch, not only at Generate Inputs.**
  `Run{N}/<freq>/inv.cfg` is what `mpiEminvTE2d` reads. Copying the staged file
  unchanged leaves Max iter / geps / dtx at whatever was typed when the inputs
  were generated. `apply_inversion_controls` patches them from the live widgets
  before the process starts. There is no per-frequency iteration list: one cap
  for every scale, plus `geps` so a scale can stop when GNORM has fallen.
- **ONE `Dataweightfile` for every component of a joint run.** Do not add
  `Dataweightfile_<SRC>` to "balance" the components. Cxz/Czx are weak on a
  layered model because a 1D earth has no lateral structure for them to sense -
  correct information, not an imbalance (section 4). Up-weighting them amplifies
  noise and modelling error exactly where they carry no signal. The upstream key
  is for differing acquisition NOISE between source types.
- **THERE IS NO SINGLE-ANYTHING ACTION.** Step 01 fixes the frequencies and the
  sources; every step after it acts on all of them. No broadband run, no
  broadband wavelet, no broadband wavelet DISPLAY, no per-source calibration
  button, no component subset. If a control makes the user pick one frequency,
  one source or one dataset to *act* on, delete it - a selector that only
  changes what is *displayed* is fine. This is the most frequently re-broken
  rule in the repo; it has now been re-litigated three times.
- **`sg.rss`/`ep.rss` are resampled NEAREST, not linear.** A linear resample
  makes every interface a one-cell RAMP whose medium is neither layer; nearest
  makes it a step on the grid. Measured with
  `scripts/experiments/interface_snapping.py`: transition cells per interface
  1 -> 0, and the placement error moved inside the half-cell bound in every
  dataset. `headless.build_forward_inputs` and notebook 01's `on_apply_outputs`
  both do this and MUST stay in step, or the GUI and the headless driver write
  different Earth models. Workspaces built before the change are not comparable
  with ones built after.
- **`.rss` samples are NODES, so a material boundary is at `o + (k+1/2)*d`.**
  Reading it as the "cell bottom" `o + (k+1)*d` puts every interface HALF A CELL
  too deep - measured 0.4 / 0.475 / 0.7 m at 6/4/2 kHz, against a half-cell data
  change of 4.9-6.6 % on |Hz| and a 3 % floor. `inversion_1d.blocky_layers_from_
  trace` is the one reader; `interface_snapping.py` measures the same depths
  independently and is the cross-check.
- **Candidate interfaces are snapped onto EACH FREQUENCY's own grid.**
  `analytic_1d_forward.snap_interfaces_to_grid`, through `forward_1d_gains`,
  keyed on `(eps_r, snap_dz, snap_origin_m)`. One grid for a joint fit is right
  for one tone and wrong for the rest. The evidence is the RECOVERED MODEL, not
  the true-model chi-squared, which does not separate the two - see
  `KNOWN_ISSUES.md` section 3 for both tables. The deepest interior interface is
  the pinned depth-window edge and is not fitted, so it is not snapped
  (`pin_last`).
- **`C(f)` is COMPUTED, not fitted: `C = dx*dz*s(order)`.** The engine injects
  the wavelet into one cell as `H += (dt/MU)*wav`, which represents `K*delta`
  integrated over that cell, so the moment is `wav*dx*dz`; `s = sum c_n(2n+1)`
  is the stencil consistency factor (0.999882 at order 6). Verified against the
  fitted value to **0.430 %** worst case over 6 datasets, phase <=0.081 deg
  (`scripts/experiments/analytic_C_check.py`). `scripts/modules/fd_error_model.py`
  is the one implementation. The Step 02 calibration run still happens - as a
  VALIDATION that the computed value holds, not as the source of it.
- **`sigma` is an error budget, not a fit residual.** Per frequency AND per
  receiver, relative to the datum: `(dx/r)^2` near-source spreading, `1-s`,
  half-cell interface quantisation, 0.22 % analytic quadrature. `eps_r`
  inflation contributes ZERO - it biases both sides equally. It halved the
  recovered model error (0.8417 -> 0.3569 rms log10 rho).
- **Snapping and the budget's quantisation term are ALTERNATIVES.** Snapping
  removes the error from the residual; charging for it in sigma too is
  double-counting and measured WORSE (0.9357) than snapping alone. Removing
  snapping and relying on the budget is worse still (0.9662).
  `analytic_tensor_calibration` zeroes the term when `cfg["snap_dz"]` is set.
- **The fitted sigma EXCEEDED the signal on the cross-couplings** (24-92x the
  budget, and larger than |obs| itself), because it was fitted where Hz is a
  near-null. Those components were nominally fitted and effectively were not -
  so any result that quotes an all-four-component improvement needs re-running.
  See KNOWN_ISSUES 5.
- **Every per-frequency dataset has its own grid, its own C and its own
  `n_periods_extract`.** Never read a whole band out of one per-frequency
  dataset - it only contains the tone it was designed for, and the others are
  noise.

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
