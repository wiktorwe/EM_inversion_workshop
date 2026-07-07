"""FD-design and config helpers for workshop workflows.

The FD design used to target the ADI TE2D engine at 10x the explicit CFL
limit (`k_cfl=10` in the old `recommend_dt_from_grid`) - the suite's own
`validate_layered_1d_model` shows ADI TE2D fails the layered Green's-
function check (HZ ~178% error) where the explicit engine passes (~1-4%),
so this workshop now targets the **explicit** engine exclusively
(`mpiEmmodTE2d`/`mpiEminvTE2d`). `design_explicit_fd` below replaces the
ADI-era `recommend_design`/`recommend_dt_from_grid`/`recommend_workshop_
pml_parameters` with rules built on `rockem.utils`' conductive-CFL/PML
sizing helpers (the same ones the suite's own validated examples use) - see
`scripts.modules.rockem_bridge` for how those are wired in.
"""

import re
from dataclasses import dataclass
from math import floor, pi, sqrt
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
from third_party.rockseis.io.rsfile import rsfile
from third_party.rockseis.tools.modint import run_modint

from scripts.modules.rockem_bridge import utils as rockem_utils

MU0 = 4 * pi * 1e-7
EPS0 = 8.854187817e-12
C0 = 299_792_458.0
CFG_LINE_RE = re.compile(r'^(\s*([A-Za-z0-9_]+)\s*=\s*")([^"]*)(";\s*(?:#.*)?)$')


def skin_depth(f_hz: float, mu_h_per_m: float, sigma_s_per_m: float) -> float:
    omega = 2 * pi * f_hz
    return sqrt(2.0 / (omega * mu_h_per_m * sigma_s_per_m))


@dataclass
class ExplicitDesignInputs:
    f_min_hz: float
    f_max_hz: float
    rho_min_ohm_m: float
    rho_max_ohm_m: float
    min_offset_m: float
    max_offset_m: float
    points_per_skin: int = 8
    cells_per_min_offset: int = 8
    tan_delta_floor: float = 50.0
    eps_r_cap: float = 1000.0
    sigma_max_clip_s_per_m: Optional[float] = None
    lpml: int = 13
    kappa_max: float = 11.0
    alpha_fraction_of_omega0: float = 1.0
    aperture_margin_m: float = 60.0
    dx_round_m: float = 0.05
    max_depth_offset_m: float = 0.0


@dataclass
class ExplicitDesignOutputs:
    dx_m: float
    dt_s: float
    eps_r_used: float
    explicit_cfl_safety: float
    sigma_max_s_per_m: float
    sigma_min_s_per_m: float
    eps_r_cap_binding: bool
    apertx_m: float
    lpml: int
    pml_kmax: float
    pml_smax: float
    pml_amax: float
    delta_induction_m: float
    depth_margin_m: float
    min_domain_halfdepth_m: float
    notes: str


def _safety_from_eps_r(eps_r: float) -> float:
    """`explicit_cfl_safety` vs. `eps_r`.

    Flat at the repo-wide default 0.95 for the full supported range (up to
    `eps_r_cap`'s default of 1000) - confirmed directly for the SMALL-
    offset, offset-limited-dx, off-axis-receiver regime this workshop's
    default survey sits in (offsets ~13-25 m, dx sized off the minimum
    offset rather than skin depth, rx_dz=-20 m): a properly-normalized (see
    the meta-lesson on normalized-vs-absolute comparison) FDTD-vs-analytic
    check at eps_r=1000 on this exact geometry found `explicit_cfl_safety`
    from 0.45 up to 0.98 all give the SAME ~0.7% worst-case normalized
    amplitude error (flat, not correlated with safety/dt at all) - the
    engine's own `checkStability` only rejects (loudly, `dt >= dt_cfl`) at
    safety>=1.0. No silent-wrong regime was found anywhere in that range.

    An earlier version of this function interpolated safety DOWN as eps_r
    climbed past 100 (toward 0.50 at eps_r=1000), based on a since-retracted
    finding that flat 0.95 gave a "stable but silently wrong" ~2.6x
    amplitude error at high eps_r on this workshop's own geometry - that
    finding did not reproduce under a properly-built end-to-end recheck and
    is believed to have been a comparison/normalization artifact in
    whatever produced it, not a real CFL-accuracy cliff. Do not resurrect
    the old interpolation without a fresh, independently-reproduced finding.
    """
    return 0.95


def design_explicit_fd(inputs: ExplicitDesignInputs) -> ExplicitDesignOutputs:
    """Size dx/dt/eps_r/safety/PML/aperture for the EXPLICIT TE2D engine
    from the model's resistivity range, the survey's frequency range, and
    its offsets - see module docstring. All of these are FD *numerical*
    parameters (unlike the survey geometry/wavelet, which the workshop
    keeps user-configurable): they follow deterministically from physics
    once rho_min/rho_max/f_min/f_max/offsets are fixed, so there is
    nothing left for a user to usefully hand-tune here.
    """
    if inputs.rho_min_ohm_m <= 0.0 or inputs.rho_max_ohm_m <= 0.0:
        raise ValueError("rho_min/rho_max must be positive.")
    if inputs.rho_min_ohm_m > inputs.rho_max_ohm_m:
        raise ValueError("rho_min must be <= rho_max.")
    if inputs.min_offset_m <= 0.0 or inputs.max_offset_m <= 0.0:
        raise ValueError("min_offset_m/max_offset_m must be positive.")

    sigma_max = 1.0 / inputs.rho_min_ohm_m
    if inputs.sigma_max_clip_s_per_m is not None:
        sigma_max = min(sigma_max, float(inputs.sigma_max_clip_s_per_m))
    sigma_min = 1.0 / inputs.rho_max_ohm_m

    # dx: the more restrictive of induction-based spacing (skin depth at the
    # most conductive cell, highest frequency) and offset-based spacing (the
    # near-source geometry, since a survey with sub-skin-depth offsets needs
    # a finer grid than induction alone would call for - see rockem-suite's
    # run_near_offset.py for the same "size dx off the minimum offset" idea).
    grid_out = rockem_utils.suggest_grid_size(
        rockem_utils.GridRecInputs(
            f_min_hz=inputs.f_min_hz, f_max_hz=inputs.f_max_hz,
            sigma_min_s_per_m=sigma_min, sigma_max_s_per_m=sigma_max,
            eps_r_min=7.0, eps_r_max=7.0,  # eps_r doesn't affect the induction/offset dx rule
            points_per_skin=inputs.points_per_skin, cells_per_wavelength=10,
        )
    )
    dx_induction = grid_out.delta_induction_m
    dx_offset = inputs.min_offset_m / float(inputs.cells_per_min_offset)
    dx_raw = min(dx_induction, dx_offset)
    dx = floor(dx_raw / inputs.dx_round_m) * inputs.dx_round_m
    if dx <= 0.0:
        dx = inputs.dx_round_m

    # eps_r: maximize (subject to a loss-tangent floor at the LEAST
    # conductive/most sensitive corner, f_max & sigma_min) to buy the
    # largest legitimate explicit dt - see the rockem-suite skill's
    # "raising eps_r to relax the explicit CFL limit is legitimate" gotcha.
    omega_max = 2.0 * pi * inputs.f_max_hz
    eps_r_raw = sigma_min / (inputs.tan_delta_floor * omega_max * EPS0)
    eps_r_used = float(np.clip(eps_r_raw, 7.0, inputs.eps_r_cap))
    eps_r_cap_binding = eps_r_raw > inputs.eps_r_cap

    safety = _safety_from_eps_r(eps_r_used)
    time_out = rockem_utils.suggest_time_steps(
        rockem_utils.TimeRecInputs(
            grid=rockem_utils.GridSpec(dx=dx, dz=dx), eps_r_min=eps_r_used,
            sigma_max_s_per_m=sigma_max, explicit_cfl_safety=safety, use_grid_diff_cap=False,
        )
    )
    dt = time_out.dt_explicit_cfl_s

    vmax = C0 / sqrt(eps_r_used)
    pml_out = rockem_utils.suggest_pml_parameters(
        rockem_utils.PmlRecInputs(
            lpml=inputs.lpml, dx=dx, dz=dx, f0_hz=inputs.f_max_hz, vmax_m_per_s=vmax,
            kappa_max=inputs.kappa_max, alpha_fraction_of_omega0=inputs.alpha_fraction_of_omega0,
        )
    )

    apertx = 2.0 * inputs.max_offset_m + inputs.aperture_margin_m

    # Depth margin: mirrors apertx's rule (clearance beyond the farthest
    # SURVEY feature, not a skin-depth-scaled distance - end-to-end testing
    # found the PML absorbs diffusive fields on its own conductivity, not
    # geometric standoff, so a fixed margin transfers across resistivity/
    # frequency choices the same way aperture_margin_m already does for x).
    # `max_depth_offset_m` is the largest |receiver_z - source_z| in the
    # survey (0 for the common same-depth convention) - the model must
    # extend at least this margin beyond THAT depth, in both directions
    # from the source depth, before hitting the PML - see
    # `scripts.modules.segy.pad_resistivity_for_depth_margin`, which pads
    # a loaded model that falls short of this instead of silently trusting
    # whatever depth range the SEG-Y file happened to cover.
    depth_margin = inputs.aperture_margin_m
    min_domain_halfdepth = abs(inputs.max_depth_offset_m) + depth_margin

    notes = (
        f"dx=min(induction {dx_induction:.4f} m, offset {dx_offset:.4f} m)={dx_raw:.4f} m "
        f"-> rounded down to {dx:.4f} m; eps_r={'CAPPED at' if eps_r_cap_binding else 'set to'} "
        f"{eps_r_used:.1f} (tan_delta_floor={inputs.tan_delta_floor:.0f} at f_max/sigma_min); "
        f"explicit_cfl_safety={safety:.2f}; min_domain_halfdepth={min_domain_halfdepth:.2f} m "
        f"from source depth (max_depth_offset={inputs.max_depth_offset_m:.2f} m + "
        f"margin={depth_margin:.2f} m)"
    )
    return ExplicitDesignOutputs(
        dx_m=dx, dt_s=dt, eps_r_used=eps_r_used, explicit_cfl_safety=safety,
        sigma_max_s_per_m=sigma_max, sigma_min_s_per_m=sigma_min, eps_r_cap_binding=eps_r_cap_binding,
        apertx_m=apertx, lpml=inputs.lpml, pml_kmax=pml_out.pml_kmax, pml_smax=pml_out.pml_smax,
        pml_amax=pml_out.pml_amax, delta_induction_m=dx_induction,
        depth_margin_m=depth_margin, min_domain_halfdepth_m=min_domain_halfdepth, notes=notes,
    )


def estimate_nt_and_cost(dt_s: float, rec_time_s: float, apertx_m: float, domain_depth_m: float,
                          dx_m: float, lpml: int, cost_per_cell_update_s: float = 2e-8) -> dict:
    """Rough nt/wall-time estimate for the design-check cell - see
    `ExplicitDesignOutputs.notes` for the underlying dx/dt/eps_r choices.
    `cost_per_cell_update_s` is a rule-of-thumb placeholder (order 1e-8 to
    1e-7 s/cell-update on a modern CPU core, engine- and machine-dependent)
    until measured directly on real hardware."""
    nt = int(rec_time_s / dt_s) + 1
    nx_cells = apertx_m / dx_m + 2 * lpml
    nz_cells = domain_depth_m / dx_m + 2 * lpml
    cells = nx_cells * nz_cells
    wall_s = nt * cells * cost_per_cell_update_s
    return {
        "nt": nt, "cells": cells, "wall_s_estimate": wall_s,
        "warn_heavy": nt > 3e5, "warn_very_heavy": nt > 1e6,
    }


def interpolate_rss_python(in_path, out_path, d1f=None, d3f=None, method="bspline", antialias=True):
    """Interpolate RSS data by calling vendored modint in-process."""
    in_path = Path(in_path)
    out_path = Path(out_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input RSS file not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        input=str(in_path),
        output=str(out_path),
        method=str(method),
        antialias=bool(antialias),
        verbose=False,
        d1f=(float(d1f) if d1f is not None else None),
        d2f=None,
        d3f=(float(d3f) if d3f is not None else None),
        d4f=None,
        d5f=None,
        d6f=None,
        d7f=None,
        d8f=None,
        d9f=None,
        o1f=None,
        o2f=None,
        o3f=None,
        o4f=None,
        o5f=None,
        o6f=None,
        o7f=None,
        o8f=None,
        o9f=None,
        max1=None,
        max2=None,
        max3=None,
        max4=None,
        max5=None,
        max6=None,
        max7=None,
        max8=None,
        max9=None,
    )
    rc = run_modint(args)
    if rc != 0:
        raise RuntimeError(f"modint failed with return code {rc}")


def enforce_rss_min_value(rss_path, min_value=1e-8):
    """
    Clamp RSS data in-place to a minimum value.
    Returns the number of samples that were clamped.
    """
    rss_path = Path(rss_path)
    f = rsfile()
    f.read(str(rss_path))

    data = np.asarray(f.data, dtype=np.float64)
    clamped = np.maximum(data, float(min_value))
    n_clamped = int(np.count_nonzero(clamped != data))

    dtype = np.float32 if int(f.data_format) == 4 else np.float64
    f.data = np.asarray(clamped, dtype=dtype, order="F")
    f.write(str(rss_path))
    return n_clamped


def read_cfg_values(cfg_path):
    values = {}
    for line in Path(cfg_path).read_text().splitlines():
        match = CFG_LINE_RE.match(line)
        if match:
            values[match.group(2)] = match.group(3)
    return values


def update_cfg_values(cfg_path, updates):
    cfg_path = Path(cfg_path)
    lines = cfg_path.read_text().splitlines()
    seen = set()
    out = []
    for line in lines:
        match = CFG_LINE_RE.match(line)
        if not match:
            out.append(line)
            continue
        key = match.group(2)
        if key in updates:
            out.append(f'{match.group(1)}{updates[key]}{match.group(4)}')
            seen.add(key)
        else:
            out.append(line)
    missing = sorted(set(updates.keys()) - seen)
    if missing:
        raise KeyError(f"Keys not found in {cfg_path}: {missing}")
    cfg_path.write_text("\n".join(out) + "\n")


def update_modcfg_for_workshop(
    modcfg_path,
    dtrec_s=None,
    apertx_m=None,
    aperty_m=None,
    sg_path=None,
    ep_path=None,
    an_path=None,
    wavelet_path=None,
    survey_path=None,
    pml_kmax=None,
    pml_smax=None,
    pml_amax=None,
):
    updates = {}
    if dtrec_s is not None:
        updates["dtrec"] = f"{float(dtrec_s):.6e}"
    if apertx_m is not None:
        updates["apertx"] = f"{float(apertx_m):.6f}"
    if aperty_m is not None:
        updates["aperty"] = f"{float(aperty_m):.6f}"
    if sg_path is not None:
        updates["Sg"] = str(sg_path)
    if ep_path is not None:
        updates["Ep"] = str(ep_path)
    if an_path is not None:
        updates["A"] = str(an_path)
    if wavelet_path is not None:
        updates["Wavelet"] = str(wavelet_path)
    if survey_path is not None:
        updates["Survey"] = str(survey_path)
    if pml_kmax is not None:
        updates["pml_kmax"] = f"{float(pml_kmax):.15g}"
    if pml_smax is not None:
        updates["pml_smax"] = f"{float(pml_smax):.15g}"
    if pml_amax is not None:
        updates["pml_amax"] = f"{float(pml_amax):.15g}"
    if updates:
        update_cfg_values(modcfg_path, updates)
    return read_cfg_values(modcfg_path)


__all__ = [
    "ExplicitDesignInputs",
    "ExplicitDesignOutputs",
    "design_explicit_fd",
    "estimate_nt_and_cost",
    "enforce_rss_min_value",
    "interpolate_rss_python",
    "read_cfg_values",
    "update_cfg_values",
    "update_modcfg_for_workshop",
    "skin_depth",
    "C0",
]
