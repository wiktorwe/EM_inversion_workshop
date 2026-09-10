# Numerical findings and the changes they justify

Every number here was measured on this workshop's own survey and model
(`examples/Fault_1.sgy`, 30 transmitters at z = 6050 m, colinear receivers at
-13.1 and -25.3 m, 1/2/4/6 kHz), with the scripts under
[`../scripts/experiments/`](../scripts/experiments/). Where something did **not**
improve, that is recorded too - a negative result that is written down is worth
more than one that is quietly dropped.

The FDTD-vs-analytic scale `C(f)` is the common yardstick throughout:
`C = FDTD_gain / analytic_gain`, and `|C|` should equal `dx^2` exactly (injecting
`w` into one cell represents `K*delta` integrated over that cell, so the
effective moment is `w*dx*dz`). Deviations of `|C|/dx^2` from 1, and its *drift*
across the frequency band, are the error being measured.

---

## 1. FD stencil order 2 -> 6

### The claim

rock-em's tabulated staggered first-derivative coefficients
(`lib/der/der.cpp`) are dispersion-optimised (Holberg-type), not Taylor-exact,
and do not satisfy the consistency condition `sum_n c_n (2n+1) = 1`:

| order | kappa = sum abs(w) | dt penalty | sum c_n(2n+1) | deviation |
|---|---|---|---|---|
| 2 | 1.25886 | 1.000x | 0.977010 | **-2.299 %** |
| 4 | 1.35764 | 1.078x | 0.998488 | -0.151 % |
| 6 | 1.39907 | 1.111x | 0.999882 | -0.012 % |
| 8 | 1.42313 | 1.130x | 0.999990 | -0.001 % |

At order 2 every first derivative is under-scaled by 2.3 % *regardless of dx*.
Both curls carry the factor, so the discrete Laplacian is scaled by `s^2` and the
numerical skin depth becomes `s*delta` - an error that grows with
offset/skin-depth and **cannot converge away under grid refinement**.

### The decisive test

A correlation is not a mechanism. The signature that distinguishes an
inconsistent operator from an ordinary truncation error is convergence
behaviour: halving `dx` must reduce a truncation error ~4x and must leave a
consistency error alone.

`|C|/dx^2`, mean deficit `1 - mean(|C|/dx^2)` and drift `max-min` across
1-6 kHz, all with a clean extraction window (section 4):

| | mean deficit | drift |
|---|---|---|
| **Homogeneous 1 Ohm-m** (receivers at 1.5-5.0 skin depths) | | |
| order 2, dx 0.8 m | 3.034 % | 4.191 % |
| order 2, dx 0.4 m | **3.137 %** | **4.344 %** |
| order 6, dx 0.8 m | 0.058 % | 0.022 % |
| order 6, dx 0.4 m | 0.040 % | 0.022 % |
| **Lateral-average 28 Ohm-m** (0.15-0.73 skin depths) | | |
| order 2, dx 0.8 m | 0.688 % | 0.098 % |
| order 2, dx 0.4 m | 0.319 % | 0.080 % |
| order 6, dx 0.8 m | 0.263 % | 0.056 % |
| order 6, dx 0.4 m | 0.090 % | 0.056 % |

On the diffusive model, **halving dx at order 2 makes the error very slightly
worse** while costing 8x the compute. Raising the order at the original `dx`
gives a 52x reduction for 2.1x the compute. Order 2 at `dx = 0.4 m` is 78x worse
than order 6 at `dx = 0.8 m`. Order 6 and order 8 are indistinguishable, so 6 is
where this converges; 6 also keeps the `lpml >= order + 5` margin that order 8
does not.

### Where the prediction only partly held

On the *lateral-average* model the improvement is real but modest (drift 0.098 %
-> 0.056 %), exactly as anticipated: at 0.15-0.73 skin depths the stencil error
has almost no distance over which to compound, and the field is close to
quasi-static, where a uniform rescaling of the Laplacian has little effect. The
residual there **does** converge with `dx` (0.263 % -> 0.090 % at order 6), so it
is an ordinary discretisation error, not a second consistency problem. It scales
as `(dx/r)^2`: a purpose-built survey with receivers at 6.325 m (7.9 cells)
shows `|C|/dx^2 = 0.9868`, a 1.3 % deviation against 0.3 % at 13.1/25.3 m, and
`(1/7.9)^2 = 1.6 %`. It is a function of offset-in-cells, not of frequency.

### Cost, measured

30 transmitters, `-np 6`, otherwise idle machine:

| | order 2 | order 6 | ratio |
|---|---|---|---|
| time steps | 177,463 | 197,228 | 1.111x (the kappa/CFL term) |
| wall clock | 294.2 s | 621.9 s | **2.11x** |
| per time step | | | 1.90x (the wider stencil alone) |

Not cheap. The value is that it removes a known systematic that grid refinement
cannot touch, and it matters far more as soon as anyone pushes to longer offsets
or lower frequencies - which is the whole point of UDAR.

---

## 2. The extraction window was contaminating every calibration

The source is a ramped CW wavelet (`Ramp_sqw`) whose ramp lasts `alpha / f_min`
seconds - half a period of the lowest tone at `alpha = 0.5`.
`steady_state_phasor` analyses only the **last** `n_periods` periods precisely so
that this transient is excluded.

Step 01 was writing `n_periods_extract` equal to the wavelet's own `n_periods`,
which asks for the *whole record* and defeats that protection. Re-extracting the
same FDTD traces at different window lengths (order 6, homogeneous Earth):

| `n_periods_extract` | drift in `|C|/dx^2` across 1-6 kHz |
|---|---|
| 3 | 0.023 % |
| 4 | 0.022 % |
| 4.5 | 0.538 % |
| 5 (= the record) | 0.230 % |

A factor of ten, and it was masquerading as a residual physics error - most of
what first looked like an unexplained frequency drift at order 6 was this. The
non-integer 4.5 is worse still, because it holds a non-integer number of periods
of some tones and leaks between them.

Step 01 now writes `n_periods_extract_safe = n_periods - ceil(alpha)` (4 for the
shipped defaults), and `steady_state_gains` warns if it is ever asked for a
window as long as the record.

**Consequence for existing runs:** `setup_metadata.json` will now say 4.0 rather
than 5.0 and `C(f)` shifts accordingly. That is a correction, not a regression,
but 1D inversion runs calibrated against the old value are not directly
comparable to new ones.

---

## 2b. THE LARGEST ERROR FOUND: the production data and its calibration were on different time bases

Found by asking a question that is easy to skip: **does the TRUE model fit the
data?** It is a synthetic study, so the answer is checkable, and it was no.

On transmitter 0 (78 m from the fault, so the Earth beneath it is essentially
1D), the true 5-layer model gave `chi2 = 381.8` - worse than the models the
inversion was finding (`chi2 ~ 317`). A true model that fits *worse* than an
inverted one is not a convergence problem; it means a systematic forward
mismatch far larger than the assumed noise.

The residuals were almost pure PHASE - amplitudes agreed to 0.3 % - and the
phase error grew linearly with frequency:

| f | phase error, before | predicted by a 9.99 us time offset |
|---|---|---|
| 1 kHz | -3.54 deg | -3.60 deg |
| 2 kHz | -7.29 deg | -7.19 deg |
| 4 kHz | -14.22 deg | -14.39 deg |
| 6 kHz | **+151.5 deg** | -21.6 deg, plus ~173 deg of aliasing |

Linear-in-frequency phase is the signature of a time shift. Two causes, both in
`fd_visualization.steady_state_gains`:

1. **Different time bases.** The calibration runs write `dtrec = dt_model`
   (2.535e-08 s), so their shot record has exactly the same length as the
   wavelet - 196,833 samples, 4.990010e-03 s - and the phase reference cancels
   exactly. The PRODUCTION run writes `dtrec = 1e-05` s, giving a 500-sample,
   5.000000e-03 s record against the same 4.990010e-03 s wavelet.
   `steady_state_phasor` analyses "the last N samples" and references phase to
   the FIRST sample of that window, so the two windows started 9.99 us apart -
   one `dtrec`. This does NOT cancel in the trace/wavelet ratio, and C(f) cannot
   absorb it, because C is fitted on the calibration runs where the offset is
   identically zero.

2. **Period quantisation at 6 kHz.** `steady_state_phasor` sets its window from
   `round(1/f/dt)` samples per period. At `dtrec = 1e-05` and 6 kHz that is
   `round(16.67) = 17`, stretching the window by 2 % and adding ~173 deg.

Fixed by windowing both series on the same ABSOLUTE interval, and then applying
the exact residual sub-sample correction `exp(-i*2*pi*f*(t0_trace - t0_wavelet))`
- the two windows can still only start on their own sample grids, and that
remainder is a KNOWN offset, so it is corrected rather than tolerated.

| | true-model chi2 on tx0 |
|---|---|
| before | 381.8 |
| after window alignment only | 22.2 |
| after the exact sub-sample correction | **0.028** |

A factor of 13,600, and the residual phase is now <= 0.17 deg at every
frequency. `C(f)` is unchanged to five digits, confirming the fix is confined to
the production-data path.

**What this invalidates.** Any 1D inversion result produced before this fix -
including the chi-squared values of 1.90-9.99 quoted as the benchmark to beat -
was fitting phases with a systematic error of up to 21.6 deg (and a 173 deg
aliasing artifact at 6 kHz). The 2D FWI is NOT affected: it fits time-domain
traces directly and never goes through the phasor extraction.

**Why it hid for so long.** Every check that could have caught it was run on the
calibration geometry, where `dtrec = dt_model` makes the offset exactly zero.
The bug lived only on the path from production data to the 1D inversion, and it
took comparing against the known truth - not against another model - to expose
it.

---

## 3. A half-cell depth-axis disagreement between two readers

`fdtd_analytic_calibration._read_rss_conductivity_xz` placed `.rss` samples at
`o + (k+0.5)*d`, while `rss_model.read_rss_model` - the reader notebooks 04/06
use on **the same files** - uses `o + k*d`. The engine's own coordinate mapping
(`Geometry2D::makeMap`: `res = (coord - o)/d; pos = floor(res)`) settles it: a
coordinate equal to `o` maps onto sample 0, so samples sit at `o + k*d`. Fixed in
both places.

Half a cell is 0.4 m here, which is **not** negligible: see section 6.

A visible side effect: the lateral-average calibration Earth is now sampled at
the true depths, so it resolves as 21 layers with mean 29.85 Ohm-m rather than
15 layers with mean 28.17 Ohm-m. Both are approximations of the same laterally
averaged profile (averaging across the fault throw produces a smooth ramp, and
how many distinct blocky layers it collapses into depends on the sampling); the
corrected one is the one that samples where the model actually is.

---

## 4. The analytic solver's kx quadrature, and a trap in its own checker

The kx integral is a fixed Gauss-Legendre rule on `[0, lam_max]`. It has two
independent failure modes, and only one is visible to an `n_nodes` sweep:
truncation converges cleanly and *wrongly* in `n_nodes`, because more nodes only
refine a range that was already too short.

`analytic_1d_forward.check_kx_convergence` doubled **only** `n_nodes`, making it
structurally incapable of detecting the error it was written to guard against.
Measured on the workshop's own 1D prior against an 8x reference:

| n_nodes | lam_max x | worst error |
|---|---|---|
| 120 | 1 | **1.69 %** (shipped default) |
| 120 | 2 | 0.22 % |
| 240 | 2 | 0.22 % (doubling n_nodes alone: **no change**) |
| 240 | 4 | 0.004 % |

### Why 2x and not 4x

Accuracy is worthless on a model the solver cannot evaluate. Larger `lam_max`
makes rockem-suite's documented silent-overflow mode much more likely, because
the layer matrix carries `exp(+-gamma*thickness)` with `gamma ~ lam_max`.
Over 400 draws from the workshop's own prior (5 layers, 1-150 Ohm-m, 5-50 m):

| multiplier | evaluated | rejected (guarded) | LinAlgError (hard fail) |
|---|---|---|---|
| 1 | 392 | 8 | 0 |
| **2 (adopted)** | **392** | **8** | **0** |
| 3 | 376 | 13 | 11 |
| 4 | 301 | 34 | **65 (16 % of the prior)** |

At 4x a quarter of the prior becomes unevaluable, which would cripple a
differential-evolution search far more than a 0.2 % forward error ever could.
`LinAlgError` is now also converted to a proper `ForwardRejected`, per
rockem-suite's warning that this solver must be guarded inside any automated
search.

The 15-layer lateral-average calibration Earth was already converged at
multiplier 1 (0.000 % change), so `C(f)` is bit-identical either way - the fix
helps only the inversion's thin, high-contrast candidate models.

---

## 5. The empymod line-source sign is a derivable convention, not a fudge

`empymod_line_source._EMPY_LINE_SIGN = -1.0` was justified only as "matches in
amplitude but with a uniform pi phase flip". Derived:

- rockem codes Faraday as `mu dH/dt = -curl E + K`, i.e.
  `curl E + i*omega*mu*H = +K`;
- empymod follows Hunziker et al. (2015), `curl E + z_hat*H = -J^m`, and carries
  the matching relation in its own source (`utils.check_ab`:
  `G^mm_ab(s,r,e,z) = -G^ee_ab(s,r,-z,-e)`).

So `K = -J^m` and the two differ by exactly the real factor -1.

Measured on complex values (an amplitude-only comparison is invariant under
conjugation and cannot tell a sign convention from a time convention), with a
converged y-quadrature: the ratio is `-1.000011 - 0.000002j`, flat in frequency,
offset, receiver depth and component. The conjugate ratio is nowhere near -1, so
it is not a time-convention flip. The earlier "amplitude match with a phase flip"
reading was a quadrature artifact: at the default `n_y = 120` the near-offset
integrand is under-resolved by up to 14 %.

---

## 6. Interface quantisation: a bias `C(f)` cannot absorb

The 1D inversion fits a **continuous**-interface analytic forward against FDTD
data whose interfaces are quantised onto the FD grid. `C` is one complex number
per frequency shared by every transmitter; this error is model- and
depth-dependent, so `C` cannot absorb it.

Sensitivity, measured by moving one interface of a representative 2/25/100
Ohm-m model by half a cell of that dataset's own grid:

| dataset | half a cell | max change in \|Hx\| | max change in \|Hz\| |
|---|---|---|---|
| 2 kHz (dx 1.40 m) | 0.70 m | 0.53 % | **6.58 %** |
| 4 kHz (dx 0.95 m) | 0.475 m | 1.61 % | **5.27 %** |
| 6 kHz (dx 0.80 m) | 0.40 m | 2.34 % | **4.95 %** |

against a 3 % uncertainty floor. So this is above the floor on `|Hz|` at every
tone, and worth fixing rather than only quantifying.

**Three things were wrong; all three are now fixed and measured.**

1. **The model was resampled with LINEAR interpolation**, so an interface was a
   one-cell RAMP - the FD medium at an interface was neither layer, which is not
   what a layered analytic forward models. `sg.rss`/`ep.rss` are now resampled
   with `nearest` (`headless.build_forward_inputs`, and notebook 01's
   `on_apply_outputs` - the two must stay in step). Measured with
   `scripts/experiments/interface_snapping.py`: transition cells per interface
   1 -> 0, and the placement error moved inside the half-cell bound in every
   dataset (max 0.400 / 0.375 / 0.300 m at 2/4/6 kHz) where the linear model
   exceeded it (0.500 m against a 0.400 m bound).

2. **The true-model reader placed every interface half a cell too deep.** It
   treated `.rss` samples as cell tops (`z0 + (k+1)*dz`); they are NODES, so a
   material boundary sits at `z0 + (k+1/2)*dz`. The error was exactly `dz/2` in
   every dataset - 0.4 / 0.475 / 0.7 m at 6/4/2 kHz. Given the table above, that
   was a real bias in every true-model overlay, not a cosmetic offset.
   `inversion_1d.blocky_layers_from_trace` is now the one reader, cross-checked
   against `interface_snapping.py`, which reads the same depths out of `sg.rss`
   independently.

3. **Snapping was off, and would have been to the wrong grid.** Candidate
   interfaces are now snapped onto **each frequency's own** grid
   (`analytic_1d_forward.snap_interfaces_to_grid`, through `forward_1d_gains`,
   enabled by default in notebook 05). Per frequency is not a refinement: each
   dataset resamples the same `sg.rss` on its own `dx`, so the same true
   interface at 6020.5 m lands at 6020.1 / 6020.875 / 6020.8 m in the 2/4/6 kHz
   models. One grid would be right for one tone and wrong for the other two.

Measured on the TRUE model over 5 Tx and all four tensor components
(`scripts/experiments/true_model_check.py`), against a matrix calibrated with
`lateral_average_true`:

| | reduced chi2 |
|---|---|
| no snapping | 0.0660 |
| snapped per frequency | **0.0296** |

The deepest interior interface in the inversion's parameterisation is the pinned
depth-window edge - identical for every candidate, not fitted - so it is not
snapped (`pin_last`). That, and the layered solver's requirement that the finite
stack be centred on the transmitter, are what still limit how finely a 1D model
can be described here. Neither binds this survey at chi2 0.03.

---

## 7. Artificial permittivity: real, small, and invisible to the calibration

`design_explicit_fd` inflates `eps_r` to buy a larger explicit time step, and
`analytic_1d_forward` evaluates the analytic reference at the *same* `eps_r`.
That is right for isolating discretisation error, but it means the inflation
cancels out of `C(f)` and is invisible there: FDTD and analytic can agree
perfectly while both differ from the true physics.

Measured directly, `eps_r_used = 399.4` against a physical `eps_r = 7`:

| | worst \|Hx\| bias | worst \|Hz\| bias |
|---|---|---|
| homogeneous 1 Ohm-m | 0.033 % | 0.032 % |
| lateral-average 28 Ohm-m | **0.152 %** | **0.261 %** |

The worst case is at `f_max` and `rho_max`, exactly where the loss-tangent floor
of 50 binds by construction. Raising the `eps_r` cap for the low-frequency runs
(section 9) costs only 0.03 / 0.07 %, well inside what the high-frequency runs
already accept - so it is safe by the workshop's own standard. That saving has
since been TAKEN: the cap is 5000, see section 9.

---

## 8. Deliberately not touched

- **PML.** A prior 12-case sweep put the boundary contribution below 1e-4 %. It
  is not in the error budget, and nothing here contradicted that.
- **`ROCKEM_HARMONIC_NORMAL_STAGGER`.** `ModelEmTE2D` does no staggering at all
  (Ey sits on the node), so the switch is inert on the 2D path.
- **Yee-grid staggering as an explanation for small residuals.** rockem-suite's
  gotchas record that it *is* correctly compensated in both source insertion and
  receiver recording, and explicitly rule it out for exactly this class of
  residual. Not re-litigated.
- **A narrowband / steady-state source in place of the broadband wavelet.**
  Plausibly cheaper once each run is monochromatic in purpose, but it is a second
  change; bundling it with the per-frequency split would make any difference in
  `C(f)` unattributable.


---

## 8b. 2D FWI cost, measured - and why the 2D head-to-head is NOT reported

Task 3 asks for a head-to-head between the multi-scale ladder and the current
single-stage all-frequency inversion. The 1D prototype was run (section 11); the
2D one was not, and here is the measurement behind that decision rather than a
guess.

One L-BFGS **function evaluation** of `mpiEminvTE2d` on the 1 kHz stage - the
CHEAPEST stage, on the coarsest 1.60 m grid, 30 shots at `-np 6`:

    inv.cfg written        07:22:34
    first evaluation done  07:29:00
    => 6.4 minutes per function evaluation

An iteration costs one gradient evaluation plus up to `max_linesearch = 5` more,
typically about two. So:

| | evaluations | wall clock |
|---|---|---|
| 20 iterations, 1 kHz stage | ~40 | **~4.3 h** |
| 20 iterations, 6 kHz stage (forward is 1.51x dearer) | ~40 | ~6.4 h |
| full ladder (4 stages + joint) + the single-stage baseline | | **~1.5-2 days** |

A truncated run would not answer the question - the whole point of the
comparison is where each strategy CONVERGES - so reporting one as if it were the
head-to-head would be worse than reporting nothing. What is delivered instead is
the complete staging machinery with its structural pieces verified: the
log-resistivity grid handoff (no overshoot; 0.0575-decade round-trip loss on the
true model, which is the intended long-wavelength truncation), the skin-depth
knot schedule (14.70 / 10.39 / 7.35 / 6.00 m), the regularisation schedule, and
five stages staged with each frequency's own grid, order, PML and aperture.

---

## 9. Per-frequency modelling instead of one broadband run

Two design rules pull in opposite directions: spatial sampling is set by the
HIGHEST frequency (dx must resolve the smallest skin depth) and record length by
the LOWEST (`n_periods_extract` periods of the slowest tone). Modelling the band
together applies the fine grid of 6 kHz through the long record of 1 kHz.

Predicted from `design_explicit_fd` (`per_frequency_cost.py`): 2.11x fewer
cell-steps serially, up to 6.00x less wall clock concurrently.

**Measured** on the real 30-transmitter production survey
(`per_frequency_production.py`, order 6):

| f | dx | nt | wall clock |
|---|---|---|---|
| 1 kHz | 1.60 m | 62,327 | 70.8 s |
| 2 kHz | 1.40 m | 35,616 | 51.4 s |
| 4 kHz | 0.95 m | 33,903 | 83.2 s |
| 6 kHz | 0.80 m | 32,872 | 106.6 s |
| **sum of four** | | | **312.1 s** |
| single broadband run | 0.80 m | 197,228 | **621.9 s** |

**1.99x serially, up to 5.84x concurrently** - the prediction held to within 6 %.

### Does the split reproduce the broadband calibration?

| f | dx | \|C\|/dx^2 split | broadband | diff | interface shift on that grid |
|---|---|---|---|---|---|
| 1 kHz | 1.60 m | 1.01799 | 0.99768 | **+2.04 %** | 0.400 m |
| 2 kHz | 1.40 m | 0.99534 | 0.99713 | -0.18 % | 0.700 m |
| 4 kHz | 0.95 m | 0.99894 | 0.99748 | +0.15 % | 0.675 m |
| 6 kHz | 0.80 m | 0.99722 | 0.99721 | **+0.001 %** | 0.000 m |

**Not** within the existing scatter (0.39 %) at 1 kHz. The discriminator is the
CONTROL, not a monotonicity test: the 6 kHz per-frequency run uses the *same*
grid as the broadband run (dx is set by f_max in both) and reproduces it to
0.001 %, which proves the assembly is correct. So the differences are a real
dependence on the grid each frequency was designed for. Two mechanisms, both
measured: a coarser grid resolves the near-source geometry with fewer cells per
offset (8.2 cells per minimum offset at 1 kHz against 16.4 at 6 kHz, and the
placement error scales as `(dx/r)^2`), and it places the Earth's interfaces up
to half a cell differently - in EITHER direction, which is why the differences
are not monotonic in dx.

Phase agrees to 0.085 deg throughout, so the phase reference survives the split:
each run divides by its own wavelet phasor and the time base cancels. Confirmed
independently by varying the analysis window on one per-frequency run - stable to
0.0007 % at `n_periods` 2/3/4, jumping to 0.178 % at 5 (section 2).

### The eps_r cap - measured, then taken

`eps_r = sigma_min / (tan_delta_floor * omega_max * eps0)` grows as frequency
falls, so at the old `eps_r_cap = 1000` it clipped for the 1 and 2 kHz runs
(uncapped: 2397 and 1198), throwing away part of the low-frequency time-step
gain for nothing. Removing the clip costs 0.029 % / 0.069 % bias at 1 kHz -
well inside the 0.152 % / 0.261 % the 6 kHz run already accepts (section 7).

**The cap is now 5000** (`fd.ExplicitDesignInputs.eps_r_cap`,
`headless.SetupParams.eps_r_cap`, and the Step 01 widget), which is non-binding
across the whole workshop band while still catching a runaway design. Re-measured
after the change, at the production rho range and order 6:

| f | dx (m) | eps_r before | eps_r after | cap binding | dt gain |
|---|---|---|---|---|---|
| 1 kHz | 1.60 | 1000 (capped) | 2396.7 | was yes, now no | **1.548x** |
| 2 kHz | 1.40 | 1000 (capped) | 1198.3 | was yes, now no | **1.095x** |
| 4 kHz | 0.95 | 599.2 | 599.2 | no | 1.000x |
| 6 kHz | 0.80 | 399.4 | 399.4 | no | 1.000x |

`dx` is unchanged at every frequency (asserted in the check, not assumed) - the
cap only ever touched `dt`. The uncapped values reproduce the 2397 / 1198
predicted above exactly, which is the confirmation that the clip was the only
thing binding.

Existing workspaces are unaffected: they carry their own `eps_r_used` in
`setup_metadata.json` and their own calibration. But a workspace built after this
change is NOT numerically comparable with one built before it at 1 or 2 kHz -
different `dt`, so a different `C`.

---

## 10. Kz source: the 2x2 magnetic coupling matrix

`source_type = 5` (Kz) alongside the existing `source_type = 3` (Kx) completes

| | records Hx | records Hz |
|---|---|---|
| **Kx source** | `Cxx` coaxial, strong | `Cxz` cross |
| **Kz source** | `Czx` cross | `Czz` transverse, strong |

### Normalisation - the falsifiable check

If the two source types needed different `C`, that would be a source-
normalisation bug, not physics: both inject through the same `dt/MU` coefficient
into one cell. Measured on the homogeneous Earth:

| f | \|C_Kx\|/dx^2 | \|C_Kz\|/dx^2 | \|Kz/Kx\| | arg(Kz/Kx) |
|---|---|---|---|---|
| 1 kHz | 0.99953 | 0.99934 | 0.99981 | +0.003 deg |
| 6 kHz | 0.99931 | 0.99826 | 0.99895 | -0.099 deg |

Same `C` to within 0.1 % and 0.1 deg. A second, independent sign that the Kz
path is wired correctly: the residual scatter FLIPS between components exactly as
it should - for Kx the *Hz* residual is large (5.4 % at 1 kHz) because Hz is the
near-null cross component, and for Kz the *Hx* residual is large (5.8 %) for the
identical reason.

### Reciprocity on the layered Earth

Lorentz reciprocity gives `Hz(B | Kx@A) = Hx(A | Kz@B)`. At zero depth offset,
combining it with the odd-in-offset parity of both cross components gives a
relation between quantities measured at the SAME source position:

    Hz_from_Kx(+d) = -Hx_from_Kz(+d)   =>   S = Cxz + Czx = 0

identically, in any laterally invariant Earth. Verified analytically to 5e-16 at
dz = 0 - and it correctly FAILS (ratio ~2) as soon as the receiver is at a
different depth, since the parity step needs the depth symmetry. This is specific
to the colinear geometry.

Measured in the FDTD on a purpose-built symmetric receiver line:

| | vs the cross term | vs \|Hx(Kx)\| co-component | \|cross\|/\|co\| |
|---|---|---|---|
| 1 kHz, 6.3 m | 7.9 % | **0.012 %** | 0.15 % |
| 1 kHz, 25.3 m | 12.9 % | **1.06 %** | 8.2 % |
| 6 kHz, 6.3 m | 0.56 % | **0.002 %** | 0.34 % |
| 6 kHz, 25.3 m | 1.13 % | **0.21 %** | 18.8 % |

The FDTD satisfies the identity to 2e-5 - 1e-2 of the co-component. The large
"12.9 %" is entirely an artifact of dividing by a quantity that is itself only
8 % of the signal - the cross term IS the near-null. The co-component
normalisation is the meaningful one, and it is the noise floor any fault-induced
asymmetry has to beat.

### How far ahead each coupling actually sees

Fault model minus laterally invariant reference, same survey, grid, wavelet and
FD design; departure measured against the calibration scatter for that
component; `apertx = 400 m` so the whole 160 m transmitter range is inside the
modelled domain.

Largest tool-to-fault distance at which the observable leaves its no-fault value
by more than the threshold:

| f | Cxx (coaxial) | Cxz | **Czx** (needs Kz) | Czz | S = Cxz+Czx |
|---|---|---|---|---|---|
| 1 kHz | 16 m | 76 m | **112 m** | - | 52 m |
| 2 kHz | 22 m | 64 m | **82 m** | - | 46 m |
| 4 kHz | 22 m | 52 m | **70 m** | - | 40 m |
| 6 kHz | 22 m | 46 m | **64 m** | - | 40 m |

**The cross-couplings see 2.9x to 7.0x further ahead than the direct coaxial
coupling**, and the best of them, `Czx`, exists only because the Kz source was
added. That is the answer to "does Hz make the method see further ahead": yes,
and by a measured factor, largest at low frequency.

The reference model's cross-term null `|S|/|Cxx|` measures 3.0e-07 to 1.9e-06,
i.e. the FDTD reciprocity floor - which is what makes the cross terms sensitive:
their background is essentially zero, so a small absolute perturbation is an
enormous relative change (max departures of 1300-8200 % against 3-8 % for Cxx).

`Czz`, the transverse co-coupling, never crosses its threshold (max departure
0.94-1.64 % against 2.65 %). The extra Kz run earns its keep through the CROSS
term, not through the strong transverse coupling.

**Caveat, and it matters for how to read the table.** `Cxz` and `Czx` are the
same physical quantity in a layered Earth (reciprocity), and they differ here
only because of lateral structure - yet `Czx` appears to detect ~40 % further.
Most of that gap is not physics: `Czx` is measured on **Hx**, whose calibration
scatter is 0.064 %, while `Cxz` is measured on **Hz**, whose scatter is 2.649 % -
a 41x looser threshold, because Hz from Kx at zero depth offset IS the near-null
and its relative scatter is correspondingly large. The transferable conclusion is
that the cross-coupling is the sensitive observable, and it should be read on
whichever receiver component is better calibrated.

A prediction I made from the first two frequencies and then falsified: at 1 and
2 kHz the `Czx` distance was 1.33 and 1.37 skin depths, so I predicted a constant
~1.35 delta, i.e. 57 m at 4 kHz and 47 m at 6 kHz. Measured: **70 m and 64 m**.
The distance falls with frequency but MORE SLOWLY than skin depth - 1.33, 1.37,
1.66, 1.86 delta - because the crossing point depends on the departure's
amplitude as well as its decay rate, and the amplitude itself changes with
frequency. Skin depth alone does not predict the look-ahead range.

### The look-ahead measurement, and the artifact that nearly replaced it

The first attempt reported "cross terms detect the fault 54-59 m ahead, direct
couplings only 6-25 m". **That was the modelling aperture, not the physics.**

`apertx > 0` is a source-centred TOTAL width, so a fault further than `apertx/2`
from a transmitter is not in that shot's local model at all. The workshop's
default `apertx = 2*max_offset + margin = 110.6 m` gives a 55.3 m half-width, and
the observables were BIT-IDENTICAL at the transmitters 78.0, 73.2, 68.4 and
63.6 m ahead of the fault (`|S|/|Cxx| = 1.452e-04` at all four), only starting to
move at 58.8 / 54.0 m - exactly the aperture edge. A physical response varies
continuously with distance; a constant that switches on is a domain boundary.

`stage_fault` now warns when the farthest transmitters show no variation, so this
cannot recur silently. The corrected experiment (`lookahead.py`) fixes three
things: `apertx` sized from the look-ahead range rather than the survey offsets,
an extended transmitter line, and - most importantly - a background that is a
**second model** (`examples/Fault_1_nofault.sgy`, laterally invariant, built from
the fault model's own left-side column) rather than a far transmitter. Taking the
background from far transmitters is circular, because "far enough that the signal
has decayed" is exactly the quantity being measured.


---

## 11. Multi-scale frequency ladder, prototyped in 1D

Settled in 1D first, as the task asks, because each inversion costs seconds
there and hundreds of trials are affordable. 11 transmitters spanning the fault,
3 seeds each, judged on TWO criteria: data misfit over ALL four frequencies (not
just the last stage's) and model error against the KNOWN truth.

**These numbers are from AFTER the time-base fix of section 2b.** Before it the
same experiment gave chi-squared of 317-584 and model errors of 1.11-1.17, i.e.
it was measuring which strategy best accommodated a broken phase. That run is
discarded.

| strategy | chi2 (all 4 frequencies) | model error (log10) | wall clock |
|---|---|---|---|
| **single-stage, all frequencies** | **5.080** | **0.6575** | 35.4 s |
| ladder + final joint stage | 5.382 | 0.6887 | 23.8 s |
| cumulative | 5.495 | 0.6914 | 28.6 s |
| strictly sequential ladder | 7.634 | 0.7992 | 12.1 s |

**The current single-stage inversion wins on both criteria.** Multi-scale is not
automatically better, and here it is not better at all - only cheaper.

The stage-by-stage trajectories say why, and they DISAGREE between variants,
which is the useful part:

| after stage | sequential ladder | cumulative |
|---|---|---|
| 1 kHz | chi2 35.8, err 0.804 | chi2 35.8, err 0.804 |
| 2 kHz | chi2 15.2, err 0.782 | chi2 16.4, err **0.770** |
| 4 kHz | chi2 8.7, err 0.792 | chi2 7.5, err **0.731** |
| 6 kHz | chi2 7.6, err 0.799 | chi2 5.5, err **0.691** |

The sequential ladder improves its FIT monotonically while its MODEL error stays
flat - after the first stage it stops learning about the Earth and only learns
about the data. The cumulative form is the only variant whose model error
improves monotonically, which is the behaviour the ladder is supposed to have.

This is consistent with the task's own caution, and worth stating plainly: at
0.15-0.73 skin depths there is no cycle skipping for a ladder to avoid, so its
classical justification has nothing to bite on. What remains is cost -
`ladder_joint` is 33 % cheaper than single-stage for 5 % worse chi2 and 4.7 %
worse model error - and that trade only becomes attractive once the survey moves
to true UDAR offsets, where the cycle-skipping argument becomes real.

Per transmitter the ordering does not reverse over the fault (tx 16-18): every
strategy does BETTER there (model error 0.51-0.72) than away from it
(0.78-0.80), and `single` is best or tied nearly everywhere.

### A tolerance that made the check useless in both directions

`check_kx_convergence`'s inherited `rel_tol = 1e-4` suited the OLD n_nodes-only
test, whose leg returns ~1e-10 - so it passed unconditionally and told you
nothing. Carrying that tolerance over to the new lam_max leg, which legitimately
sits near 1e-3 (measured 1.4e-3 over this workshop's prior at the shipped
quadrature policy), made it FAIL unconditionally instead. Same uselessness, other
direction, and it is what a user hit: the panel always said "NOT CONVERGED".

`rel_tol` is now anchored to the workshop's own uncertainty floor - a tenth of
`VALIDATED_REL_ERROR_FLOOR` (3 %), i.e. 0.3 % - and the verdict has three levels
rather than two, so "inside the noise floor but not negligible" is visible
instead of being rounded to pass or fail. The shipped configuration reports
`converged` with real margin (1.4e-3 against a 3e-3 tolerance), and a prior wide
enough to matter still trips it.

The lesson generalises: a pass/fail threshold inherited from a different test is
worse than no threshold, because it looks like a measurement.

### Recommendation

Do not adopt a strictly sequential ladder. If the ladder is wanted for cost, use
the **cumulative** form or the sequential ladder **with a final joint stage** -
both land within 5 % of the single-stage result. Re-test once offsets reach
several skin depths, where the conditioning argument should finally pay.
