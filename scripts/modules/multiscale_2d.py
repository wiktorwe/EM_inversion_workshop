"""Multi-scale 2D FWI: a frequency ladder over the per-frequency grids of Task 2.

Once Task 2 splits the modelling by frequency, the inversion HAS to follow: each
frequency owns a dataset recorded on its own grid with its own dx and dt, so a
single inversion consuming the whole band no longer has one coherent set of
inputs. This module turns that necessity into the better-conditioned approach -
an outer loop over frequency, lowest first, each stage initialised from the
previous stage's model.

What a stage is
---------------
Stage k inverts the dataset for frequency f_k, ON THE GRID Task 2 designed for
f_k, with `inv.cfg` values matched to that stage:

* `order` and `lpml` copied from that frequency's forward `mod.cfg` (this is
  what `prepare_inversion_inputs` already does - a mismatch would make the FWI's
  re-modelled wavefield inconsistent with the data it is fitting);
* that run's PML parameters, likewise;
* that run's `Recordfile_HX` / `Recordfile_HZ`;
* `Sg` = the previous stage's OUTPUT model, resampled onto this stage's grid;
* `dtx`/`dtz` (the B-spline knot spacing, `paramtype=1`) scaled with skin
  depth, so early stages solve for genuinely fewer, larger-scale parameters;
* `tik_sgregalpha` / `tv_sgregalpha` scheduled to relax as frequency rises.

Stage 1 starts from whatever `create_initial_sg0_model` produces today.

The grid handoff
----------------
This is the new structural piece and the one most likely to be got wrong: stage
k's output lives on stage k's grid and stage k+1 needs it on a finer one.

`resample_model_log_rho` interpolates in **log resistivity** - not in
conductivity and not in resistivity - so that a smooth coarse model does not
develop overshoot at contrasts, and writes the result out as an inspectable
`.rss` rather than resampling inside a pipeline where nobody can look at it.
`verify_roundtrip` checks the operator on the KNOWN true model before it is
trusted on an inverted one.

Why log resistivity, measured rather than asserted. Halfway across the fault
model's own 2 -> 100 Ohm-m contrast, a linear interpolant returns:

    interpolating in resistivity   ->  51.00 Ohm-m   (dominated by the resistive side)
    interpolating in conductivity  ->   3.92 Ohm-m   (dominated by the conductive side)
    interpolating in log10(rho)    ->  14.14 Ohm-m   = sqrt(2*100), the geometric mean

The two "obvious" choices differ from each other by 13x on the SAME contrast,
and which one is biased high depends only on which variable you happened to
hold. The geometric mean is the neutral choice, and it is what a log-parameterised
inversion effectively assumes anyway.

Note on `overshoot`: linear interpolation is monotone, so it cannot produce a
value outside the source range in ANY variable - the flag reported below can
never fire for the current interpolant. It is kept because a future switch to a
non-monotone interpolant (the B-spline `modint` used elsewhere in this workshop,
say) CAN overshoot, and overshoot at a contrast is exactly the failure mode that
would put unphysical resistivities into the next stage's starting model.

Per-stage weighting, uncertainties and aperture
-----------------------------------------------
`inversion.create_weight_file_from_hx` builds a Hann taper of length `nt` from
THAT stage's own `Hx_data.rss`, so it is already correct per stage without
change: each per-frequency run has its own record length (`rec_time =
n_periods / f`, so the high-frequency records are shorter), and the weight file
inherits it. The taper also happens to suppress the source ramp-up at the start
of the record, which is the same transient `n_periods_extract` has to avoid on
the phasor-extraction side.

`apertx` is taken from each stage's own forward `setup_metadata.json`, not from
the `inv.cfg` template. The template's `apertx = "60"` is a fallback only -
notebook 03's widget already defaults to the forward run's `apertx_m` (110.6 m
for the shipped survey), and `apertx > 0` is a source-centred TOTAL width, so 60
would put the 25.3 m receivers only 4.7 m clear of the PML instead of the 60 m
the forward design asked for. Propagating the forward value per stage is the
correct choice and is what `build_ladder` does.

Sigmas do not enter the 2D path the way they do in 1D: the engine weights data
through `Dataweightfile`, not through a per-frequency sigma array. The
per-frequency calibration scatter is still worth carrying because it is what
tells you when a stage has fit to the noise floor - it is the stopping criterion,
not an input.

Regularisation schedule
-----------------------
A multi-scale ladder without one has a specific failure mode: the low-frequency
stages converge to a smooth model, and if the penalty is not relaxed as
frequency rises the later stages cannot add the detail they exist to add, so the
result is WORSE than the single-stage inversion it replaced. Start heavier and
relax - `stage_regularisation` implements that as a power law in f_max/f_k, with
the final stage anchored at whatever value the single-stage run uses, so the
ladder and the baseline agree at the top of the band by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from third_party.rockseis.io.rsfile import rsfile


# ---------------------------------------------------------------------------
# grid handoff
# ---------------------------------------------------------------------------
def read_sg_grid(path: Path | str) -> dict:
    """(x, z, sigma[nx,nz]) plus the raw geometry, from a conductivity .rss."""
    f = rsfile()
    f.read(str(path))
    data = np.asarray(f.data, dtype=float)
    if data.ndim == 3:
        data = data[:, data.shape[1] // 2, :]
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"expected a 2D conductivity model at {path}, got {np.shape(f.data)}")
    nx, nz = data.shape
    dx, ox = float(f.geomD[0]), float(f.geomO[0])
    iz = 2 if (len(f.geomN) > 2 and int(f.geomN[2]) > 0) else 1
    dz, oz = float(f.geomD[iz]), float(f.geomO[iz])
    return {
        "sigma": data,
        # samples sit AT o + k*d (the engine's own convention - see
        # fdtd_analytic_calibration._read_rss_conductivity_xz), not at cell
        # centres
        "x": ox + np.arange(nx, dtype=float) * dx,
        "z": oz + np.arange(nz, dtype=float) * dz,
        "dx": dx, "dz": dz, "ox": ox, "oz": oz, "nx": nx, "nz": nz,
    }


def resample_model_log_rho(src_path: Path | str, template_path: Path | str,
                           out_path: Path | str) -> dict:
    """Put `src_path`'s model onto `template_path`'s grid, via log10(rho).

    `template_path` is that stage's own `sg.rss`: it defines the exact target
    grid (origin, spacing, extent), so the resampled model is guaranteed to be
    something the engine can read alongside the rest of that stage's inputs.

    Values outside the source extent are held at the nearest edge value rather
    than extrapolated - extrapolating log-resistivity off the end of an inverted
    model produces confident nonsense in exactly the region the data never
    constrained.

    Returns a small report (min/max resistivity before and after, and the
    resampling error measured by mapping straight back) so the handoff can be
    checked rather than assumed.
    """
    src = read_sg_grid(src_path)
    tpl = read_sg_grid(template_path)

    log_rho = -np.log10(np.clip(src["sigma"], 1e-30, None))

    def interp2(field, sx, sz, tx, tz):
        # separable linear interpolation with edge hold
        out = np.empty((tx.size, sz.size), dtype=float)
        for k in range(sz.size):
            out[:, k] = np.interp(tx, sx, field[:, k], left=field[0, k], right=field[-1, k])
        out2 = np.empty((tx.size, tz.size), dtype=float)
        for i in range(tx.size):
            out2[i, :] = np.interp(tz, sz, out[i, :], left=out[i, 0], right=out[i, -1])
        return out2

    log_rho_t = interp2(log_rho, src["x"], src["z"], tpl["x"], tpl["z"])
    sigma_t = np.power(10.0, -log_rho_t)

    f = rsfile()
    f.read(str(template_path))
    arr = np.asarray(f.data)
    f.data = np.asfortranarray(sigma_t.reshape(arr.shape).astype(arr.dtype, copy=False))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    f.write(str(out_path))

    back = interp2(log_rho_t, tpl["x"], tpl["z"], src["x"], src["z"])
    return {
        "src_dx": src["dx"], "src_dz": src["dz"], "dst_dx": tpl["dx"], "dst_dz": tpl["dz"],
        "src_shape": [src["nx"], src["nz"]], "dst_shape": [tpl["nx"], tpl["nz"]],
        "rho_min_src": float(np.min(10.0 ** log_rho)), "rho_max_src": float(np.max(10.0 ** log_rho)),
        "rho_min_dst": float(np.min(10.0 ** log_rho_t)), "rho_max_dst": float(np.max(10.0 ** log_rho_t)),
        "overshoot": bool(np.min(log_rho_t) < np.min(log_rho) - 1e-9
                          or np.max(log_rho_t) > np.max(log_rho) + 1e-9),
        "roundtrip_rms_log10": float(np.sqrt(np.mean((back - log_rho) ** 2))),
        "roundtrip_max_log10": float(np.max(np.abs(back - log_rho))),
        "out": str(out_path),
    }


def verify_roundtrip(true_sg: Path | str, coarse_template: Path | str,
                     fine_template: Path | str, work_dir: Path | str) -> dict:
    """Check the handoff operator on the KNOWN true model, coarse -> fine.

    Maps the true model down onto the coarse stage grid and back up onto the
    fine one, and reports how much of it survives. Two things matter and are
    reported separately:

    * `roundtrip_rms_log10` - how much detail the coarse grid cannot hold. This
      is a real, expected loss, not a bug: it is exactly the long-wavelength
      truncation the early stages are supposed to impose.
    * `overshoot` - whether the interpolant produced resistivities OUTSIDE the
      source range. Always False for the current (monotone, linear-in-log)
      interpolant; it is a tripwire for a future non-monotone one, not evidence
      that this one is correct. See the module docstring.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    down = resample_model_log_rho(true_sg, coarse_template, work_dir / "true_on_coarse.rss")
    up = resample_model_log_rho(work_dir / "true_on_coarse.rss", fine_template,
                                work_dir / "true_on_coarse_then_fine.rss")
    a = read_sg_grid(true_sg)
    b = read_sg_grid(work_dir / "true_on_coarse_then_fine.rss")
    # compare on the finer of the two, interpolating the true model onto it
    la = -np.log10(np.clip(a["sigma"], 1e-30, None))
    lb = -np.log10(np.clip(b["sigma"], 1e-30, None))
    tmp = np.empty((b["nx"], a["z"].size))
    for k in range(a["z"].size):
        tmp[:, k] = np.interp(b["x"], a["x"], la[:, k], left=la[0, k], right=la[-1, k])
    la_on_b = np.empty((b["nx"], b["nz"]))
    for i in range(b["nx"]):
        la_on_b[i, :] = np.interp(b["z"], a["z"], tmp[i, :], left=tmp[i, 0], right=tmp[i, -1])
    return {
        "down": down, "up": up,
        "true_vs_roundtrip_rms_log10": float(np.sqrt(np.mean((lb - la_on_b) ** 2))),
        "true_vs_roundtrip_max_log10": float(np.max(np.abs(lb - la_on_b))),
        "overshoot_anywhere": bool(down["overshoot"] or up["overshoot"]),
    }


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------
def stage_knot_spacing(freq_hz: float, f_max_hz: float, dt_final_m: float = 6.0,
                       exponent: float = 0.5) -> float:
    """B-spline knot spacing for one stage, scaled with skin depth.

    delta ~ 1/sqrt(f), so anchoring the FINAL stage at today's value gives
    `dt_final * (f_max/f)**0.5`: about 14.7 / 10.4 / 7.3 / 6.0 m at 1 / 2 / 4 /
    6 kHz for `dt_final = 6.0`. That is a starting point to test, not a tuned
    answer - `exponent` is exposed so it can be swept.
    """
    return float(dt_final_m) * (float(f_max_hz) / float(freq_hz)) ** float(exponent)


def stage_regularisation(freq_hz: float, f_max_hz: float, alpha_final: float,
                         exponent: float = 1.0) -> float:
    """Tikhonov/TV weight for one stage: heavier low, relaxing to `alpha_final`.

    `alpha_final` anchors the TOP of the band at whatever the single-stage
    inversion uses, so the ladder and the head-to-head baseline agree there by
    construction and the comparison isolates the ladder, not the penalty.
    """
    return float(alpha_final) * (float(f_max_hz) / float(freq_hz)) ** float(exponent)


@dataclass
class Stage:
    index: int
    freq_hz: float
    source_field: str
    forward_dir: Path
    run_dir: Path
    dx_m: float
    dtx_m: float
    dtz_m: float
    tik_sgregalpha: float
    tv_sgregalpha: float
    max_iterations: int
    apertx_m: float
    sg0_from: Optional[Path] = None
    resample_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}
        return d


def build_ladder(
    per_frequency_manifest: dict,
    ladder_root: Path | str,
    *,
    alpha_final: float,
    dt_final_m: float = 6.0,
    max_iterations: int = 20,
    knot_exponent: float = 0.5,
    reg_exponent: float = 1.0,
    joint_final_stage: bool = True,
    source_fields: Sequence[str] = ("HX",),
) -> list[Stage]:
    """Stages, lowest frequency first, plus an optional final joint stage.

    The design decision the task asks to make DELIBERATELY: whether stage k uses
    only f_k, or all frequencies up to and including f_k. The cumulative form is
    the more robust textbook choice, but it conflicts with Task 2 - frequencies
    below f_k live on COARSER grids, so a cumulative stage has to run on the
    finest grid of its set and gives back part of the modelling saving.

    The compromise implemented here is a strictly SEQUENTIAL ladder (each stage
    on its own grid, cheap, doing its classical job of producing a good starting
    model), followed by ONE final joint stage on the finest grid using all
    frequencies together, starting from the ladder's output - so the final model
    actually fits every frequency rather than only the last one.

    MULTI-SOURCE, and what it can and cannot be with this engine.
    `source_fields` cascades over source components INSIDE each frequency:
    stage order is (f0,Kx), (f0,Kz), (f1,Kx), (f1,Kz), ... each starting from the
    previous stage's model. Every dataset is therefore used, and the final model
    has seen all of them.

    That is a CASCADE, not a joint inversion, and the difference is real: a
    joint inversion sums the Kx and Kz gradients before stepping, so both
    constrain the same update; a cascade fits one source, then moves to the
    next, so the last source in each frequency has the final say.

    The reason it is a cascade is an engine constraint, not a choice.
    `mpiEminvTE2d` parses `source_type` as a single int and stores it as one
    scalar on InversionEmTE2D; the switch that applies it
    (lib/inversion/inversionEmTE2D.cpp) already sits INSIDE the per-shot loop,
    but reads that global value, so every shot in a run gets the same source
    type. A true joint multi-source inversion needs that value to become
    per-shot - a small, well-localised change, but an UPSTREAM one in
    rockem-suite, not something to bodge in the workshop.
    """
    ladder_root = Path(ladder_root)
    runs = [r for r in per_frequency_manifest["runs"].values() if r.get("freq_hz") is not None]
    if not runs:
        raise ValueError(
            "The manifest contains no per-frequency datasets. Build them with "
            "build_forward_matrix(..., split_by_frequency=True) - a broadband "
            "manifest cannot drive a frequency ladder."
        )
    sources = [str(s).upper() for s in (source_fields or ("HX",))]
    by_freq: dict[float, dict[str, dict]] = {}
    for r in runs:
        by_freq.setdefault(float(r["freq_hz"]), {})[str(r.get("source_field", "HX")).upper()] = r
    missing = {f: [s for s in sources if s not in d] for f, d in by_freq.items()}
    missing = {f: m for f, m in missing.items() if m}
    if missing:
        raise ValueError(f"Manifest is missing source datasets: {missing}")

    f_max = max(by_freq)
    stages: list[Stage] = []
    k = 0
    for f in sorted(by_freq):                      # lowest frequency first
        for src in sources:                        # then cascade over sources
            r = by_freq[f][src]
            meta = r["meta"]
            stages.append(Stage(
                index=k, freq_hz=f, source_field=src, forward_dir=Path(r["run_dir"]),
                run_dir=ladder_root / f"stage{k}_f{f:.0f}Hz_{src.lower()}",
                dx_m=float(meta["dx_model_target_m"]),
                dtx_m=stage_knot_spacing(f, f_max, dt_final_m, knot_exponent),
                dtz_m=stage_knot_spacing(f, f_max, dt_final_m, knot_exponent),
                tik_sgregalpha=stage_regularisation(f, f_max, alpha_final, reg_exponent),
                tv_sgregalpha=0.0,
                max_iterations=max_iterations,
                apertx_m=float(meta["apertx_m"]),
            ))
            k += 1
    if joint_final_stage:
        for src in sources:
            r = by_freq[f_max][src]
            meta = r["meta"]
            stages.append(Stage(
                index=len(stages), freq_hz=f_max, source_field=src,
                forward_dir=Path(r["run_dir"]),
                run_dir=ladder_root / f"stage_joint_{src.lower()}",
                dx_m=float(meta["dx_model_target_m"]),
                dtx_m=dt_final_m, dtz_m=dt_final_m,
                tik_sgregalpha=float(alpha_final), tv_sgregalpha=0.0,
                max_iterations=max_iterations, apertx_m=float(meta["apertx_m"]),
            ))
    return stages


__all__ = [
    "Stage",
    "build_ladder",
    "read_sg_grid",
    "resample_model_log_rho",
    "stage_knot_spacing",
    "stage_regularisation",
    "verify_roundtrip",
]
