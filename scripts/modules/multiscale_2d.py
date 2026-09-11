"""Multi-scale 2D FWI: a frequency ladder over the workshop's per-frequency grids.

The modelling is split by frequency, so the inversion HAS to follow: each
frequency owns a dataset recorded on its own grid with its own dx and dt, and a
single inversion consuming the whole band has no coherent set of inputs to
consume. This module turns that necessity into the better-conditioned approach -
an outer loop over frequency, lowest first, each stage initialised from the
previous stage's model.

What a stage is
---------------
Stage k inverts the dataset for frequency f_k, ON THAT FREQUENCY'S OWN GRID,
with `inv.cfg` values matched to that stage:

* `order` and `lpml` copied from that frequency's forward `mod.cfg` (this is
  what `prepare_inversion_inputs` already does - a mismatch would make the FWI's
  re-modelled wavefield inconsistent with the data it is fitting);
* that run's PML parameters, likewise;
* that run's `Recordfile_HX` / `Recordfile_HZ`;
* `Sg` = the previous stage's OUTPUT model, resampled onto this stage's grid;
* `dtx`/`dtz` (the B-spline knot spacing, `paramtype=1`) scaled with skin
  depth, so early stages solve for genuinely fewer, larger-scale parameters;
* `tik_sgregalpha` / `tv_sgregalpha` scheduled to relax as frequency rises.

Stage 1 starts from whatever `create_initial_sg0_model` produces.

The grid handoff
----------------
This is the structural piece most likely to be got wrong: stage k's output
lives on stage k's grid and stage k+1 needs it on a finer one.

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
THAT stage's own Hx record, so it is correct per stage with no extra
work: each per-frequency run has its own record length (`rec_time =
n_periods / f`, so the high-frequency records are shorter), and the weight file
inherits it. ONE weight is shared by every component of a joint stage, on
purpose - see `prepare_inversion_inputs` for why the cross-couplings are not
up-weighted to match the co-components. The taper also happens to suppress the
source ramp-up at the start of the record, which is the same transient
`n_periods_extract` has to avoid on the phasor-extraction side.

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
import re
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
    """One inversion run of the ladder.

    `source_fields` and `forward_dirs` are PLURAL because a stage can now be
    joint: `mpiEminvTE2d` takes a comma-separated `source_type`, so several
    source components go into one run and stack into one gradient. A
    single-source stage is the one-entry case of the same thing.
    """

    index: int
    freq_hz: float
    source_fields: tuple[str, ...]
    forward_dirs: dict[str, Path]
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

    @property
    def reference_dir(self) -> Path:
        """The forward run to read the grid, true model and permittivity from.

        Any of them would do: `prepare_inversion_inputs` asserts that every
        source of a stage agrees on order/lpml/PML, and they are built by
        `build_forward_matrix` from one `SetupParams` differing only in
        `source_field`, so they share `sg.rss`, `ep.rss` and the survey.
        """
        return self.forward_dirs[self.source_fields[0]]

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict) and any(isinstance(x, Path) for x in v.values()):
                d[k] = {kk: str(vv) for kk, vv in v.items()}
            elif isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
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
    source_mode: str = "joint",
) -> list[Stage]:
    """Stages, lowest frequency first, plus an optional final stage.

    The design decision to make DELIBERATELY: whether stage k uses only f_k, or
    all frequencies up to and including f_k. The cumulative form is the more
    robust textbook choice, but it conflicts with per-frequency grids -
    frequencies below f_k live on COARSER ones, so a cumulative stage has to run
    on the finest grid of its set and gives back part of the modelling saving.

    The compromise implemented here is a strictly SEQUENTIAL ladder, each stage
    on its own grid, cheap, doing its classical job of producing a good starting
    model. `joint_final_stage` then adds one more stage on the FINEST grid with
    the final knot spacing and regularisation, starting from the ladder's output.
    Note what that final stage is and is not: it re-inverts the highest frequency
    with the final settings, it does NOT fit every frequency at once. A genuinely
    all-frequency stage would need the lower tones resampled onto the fine grid
    and is not built here.

    MULTI-SOURCE. `source_fields` names the source components to invert;
    `source_mode` decides how they are combined.

    "joint" (the default) puts ALL of them in ONE inversion per frequency.
    `mpiEminvTE2d` takes a comma-separated `source_type`, so its work list spans
    (shot x source type) and every source stacks into the same gradient: the
    joint gradient is the sum of the per-source gradients, and every model update
    sees all four tensor components at once. It costs about what the separate
    single-source runs cost combined - roughly 2x one source for two of them, not
    4x - because both receiver components already come from one propagation per
    (shot, source).

    "cascade" is the older behaviour: one stage per (frequency, source), each
    starting from the previous stage's model. It is NOT forced by the engine any
    more, and it is not equivalent - a cascade lets the last source of each
    frequency have the final say, where a joint stage makes both constrain the
    same update. Keep it for two things: bisecting which source component is
    driving a result, and reproducing a ladder built before joint inversion
    existed.

    HARD REQUIREMENT for a joint stage: every record file it uses must share a
    trace count, trace order, per-trace coordinates and time axis. The engine
    builds ONE shot keymap from the first file and applies it to all of them, and
    `apertx > 0` is a source-centred TOTAL width taken from each gather's own
    coordinates, so a coordinate disagreement would silently give the same shot a
    different local model per source type. `mpiEminvTE2d` checks this at startup
    and aborts naming the file and the trace index
    (src/mpiEminvTE2d.cpp:553-585). The workshop satisfies it because
    `headless.build_forward_matrix` builds a frequency's source datasets with
    `replace(p, source_field=src, ...)` - only `source_type` differs, the survey
    is shared.
    """
    ladder_root = Path(ladder_root)
    mode = str(source_mode).lower()
    if mode not in {"joint", "cascade"}:
        raise ValueError(f"source_mode must be 'joint' or 'cascade', got {source_mode!r}")
    runs = [r for r in per_frequency_manifest["runs"].values() if r.get("freq_hz") is not None]
    if not runs:
        raise ValueError(
            "The manifest contains no per-frequency datasets. Rebuild the "
            "workspace with Step 01 / build_forward_matrix - a legacy broadband "
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

    # A joint stage covers every source at once; a cascade emits one per source.
    groups = [tuple(sources)] if mode == "joint" else [(s,) for s in sources]

    def _tag(srcs: Sequence[str]) -> str:
        return "_".join(s.lower() for s in srcs)

    f_max = max(by_freq)
    stages: list[Stage] = []
    k = 0
    for f in sorted(by_freq):                      # lowest frequency first
        for srcs in groups:
            meta = by_freq[f][srcs[0]]["meta"]
            stages.append(Stage(
                index=k, freq_hz=f, source_fields=tuple(srcs),
                forward_dirs={s: Path(by_freq[f][s]["run_dir"]) for s in srcs},
                run_dir=ladder_root / f"stage{k}_f{f:.0f}Hz_{_tag(srcs)}",
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
        for srcs in groups:
            meta = by_freq[f_max][srcs[0]]["meta"]
            stages.append(Stage(
                index=len(stages), freq_hz=f_max, source_fields=tuple(srcs),
                forward_dirs={s: Path(by_freq[f_max][s]["run_dir"]) for s in srcs},
                run_dir=ladder_root / f"stage_joint_{_tag(srcs)}",
                dx_m=float(meta["dx_model_target_m"]),
                dtx_m=dt_final_m, dtz_m=dt_final_m,
                tik_sgregalpha=float(alpha_final), tv_sgregalpha=0.0,
                max_iterations=max_iterations, apertx_m=float(meta["apertx_m"]),
            ))
    return stages


_SG_UP_RE = re.compile(r"sg_up\.rss-(\d+)$")


def find_output_model(run_dir: Path | str) -> Optional[Path]:
    """The inverted Sg the engine wrote.

    `Results/sg_up.rss-<iter>` is where `saveResults` puts every accepted
    model (`inversionBase.h:28`), so it is looked for FIRST. The globs
    below it are the fallback for run directories written by older builds.
    A ladder that cannot find this file must STOP, not continue from the
    uniform start - that is how N independent inversions masquerade as a
    frequency ladder.
    """
    run_dir = Path(run_dir)
    candidates = list(run_dir.glob("Results/sg_up.rss-*")) + list(run_dir.glob("sg_up.rss-*"))
    if candidates:
        def _suffix(path: Path) -> int:
            m = _SG_UP_RE.search(path.name)
            return int(m.group(1)) if m else -1
        return sorted(candidates, key=lambda p: (_suffix(p), p.name))[-1]
    staged = {"sg0.rss", "ep.rss", "wav2d.rss", "weight.rss",
              "Sg_min.rss", "Sg_max.rss"}
    for pattern in ("*Sg*final*.rss", "*sg*final*.rss", "*Sg_*.rss", "*sg_*.rss", "*.rss"):
        hits = sorted(
            p for p in run_dir.glob(pattern)
            if "grad" not in p.name.lower()
            and not p.name.endswith("_data.rss")
            and p.name not in staged
        )
        if hits:
            return hits[-1]
    return None


def handoff_starting_model(
    previous_model: Path | str | None,
    template_sg: Path | str,
    dest_sg0: Path | str,
) -> Optional[dict]:
    """Write `dest_sg0` from the previous inverted model, on this stage's grid.

    `previous_model is None` is the first stage: keep the uniform `sg0` already
    written by `create_initial_sg0_model` and return None. Any later stage
    MUST pass a real path; a missing file is an error, not a silent fall-back
    to the uniform start.
    """
    if previous_model is None:
        return None
    previous_model = Path(previous_model)
    if not previous_model.exists():
        raise FileNotFoundError(
            f"Previous inverted model {previous_model} is missing; "
            "the next frequency has no starting model to resample."
        )
    return resample_model_log_rho(previous_model, template_sg, dest_sg0)


__all__ = [
    "Stage",
    "build_ladder",
    "find_output_model",
    "handoff_starting_model",
    "read_sg_grid",
    "resample_model_log_rho",
    "stage_knot_spacing",
    "stage_regularisation",
    "verify_roundtrip",
]
