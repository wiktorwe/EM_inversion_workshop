"""FD-design and config helpers for workshop workflows."""

import re
from dataclasses import dataclass
from math import log, pi, sqrt
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
from third_party.rockseis.io.rsfile import rsfile
from third_party.rockseis.tools.modint import run_modint


MU0 = 4 * pi * 1e-7
EPS0 = 8.854187817e-12
C0 = 299_792_458.0
CFG_LINE_RE = re.compile(r'^(\s*([A-Za-z0-9_]+)\s*=\s*")([^"]*)(";\s*(?:#.*)?)$')


def skin_depth(f_hz: float, mu_h_per_m: float, sigma_s_per_m: float) -> float:
    omega = 2 * pi * f_hz
    return sqrt(2.0 / (omega * mu_h_per_m * sigma_s_per_m))


def recommend_grid_spacing(
    f_max_hz: float,
    sigma_max_s_per_m: float,
    mu_h_per_m: float = MU0,
    points_per_skin: int = 5,
) -> float:
    delta_min = skin_depth(f_max_hz, mu_h_per_m, sigma_max_s_per_m)
    return delta_min / float(points_per_skin)


@dataclass
class GridDtInputs:
    dx_m: float
    sigma_max_s_per_m: float
    mu_h_per_m: float = MU0
    dim: int = 3
    k_cfl: float = 10.0


@dataclass
class GridDtOutputs:
    dt_cfl_s: float
    dt_wave_cap_s: float
    dt_grid_diff_cap_s: float
    dt_adi_s: float


def recommend_dt_from_grid(i: GridDtInputs) -> GridDtOutputs:
    dt_cfl = i.dx_m / (C0 * sqrt(i.dim))
    dt_wave_cap = i.k_cfl * dt_cfl
    dt_grid_diff_cap = i.mu_h_per_m * i.sigma_max_s_per_m * i.dx_m * i.dx_m / 4.0
    dt_adi = min(dt_wave_cap, dt_grid_diff_cap)
    return GridDtOutputs(dt_cfl, dt_wave_cap, dt_grid_diff_cap, dt_adi)


@dataclass
class DesignInputs:
    f_min_hz: float
    f_max_hz: float
    sigma_min_s_per_m: float
    sigma_max_s_per_m: float
    mu_h_per_m: float = MU0
    dim: int = 3
    dx_override_m: Optional[float] = None
    points_per_skin: int = 5
    k_cfl: float = 10.0


@dataclass
class DesignOutputs:
    dx_m: float
    delta_min_m: float
    dt_cfl_s: float
    dt_wave_cap_s: float
    dt_grid_diff_cap_s: float
    dt_adi_s: float


@dataclass
class PmlHeuristicInputs:
    lpml: int
    dx_m: float
    dz_m: float
    f0_hz: float
    vmax_m_per_s: float
    target_reflection: float = 1e-6
    kappa_max: float = 11.0
    alpha_fraction_of_omega0: float = 1.0
    poly_order_m: int = 3


@dataclass
class PmlHeuristicOutputs:
    pml_amax: float
    pml_kmax: float
    pml_smax: float
    thickness_m: float


def recommend_design(inputs: DesignInputs) -> DesignOutputs:
    if inputs.dx_override_m is not None and inputs.dx_override_m > 0.0:
        dx = inputs.dx_override_m
    else:
        dx = recommend_grid_spacing(
            f_max_hz=inputs.f_max_hz,
            sigma_max_s_per_m=inputs.sigma_max_s_per_m,
            mu_h_per_m=inputs.mu_h_per_m,
            points_per_skin=inputs.points_per_skin,
        )
    delta_min = skin_depth(inputs.f_max_hz, inputs.mu_h_per_m, inputs.sigma_max_s_per_m)
    dt_out = recommend_dt_from_grid(
        GridDtInputs(
            dx_m=dx,
            sigma_max_s_per_m=inputs.sigma_max_s_per_m,
            mu_h_per_m=inputs.mu_h_per_m,
            dim=inputs.dim,
            k_cfl=inputs.k_cfl,
        )
    )
    return DesignOutputs(dx, delta_min, dt_out.dt_cfl_s, dt_out.dt_wave_cap_s, dt_out.dt_grid_diff_cap_s, dt_out.dt_adi_s)


def lossy_medium_phase_velocity(f_hz: float, mu_h_per_m: float, eps_f_per_m: float, sigma_s_per_m: float) -> float:
    omega = 2.0 * pi * float(f_hz)
    ratio = float(sigma_s_per_m) / (omega * float(eps_f_per_m))
    return sqrt(1.0 / (((float(mu_h_per_m) * float(eps_f_per_m)) / 2.0) * (sqrt(1.0 + ratio * ratio) + 1.0)))


def recommend_pml_parameters(inputs: PmlHeuristicInputs) -> PmlHeuristicOutputs:
    if inputs.lpml <= 0:
        raise ValueError("lpml must be > 0.")
    if inputs.dx_m <= 0.0 or inputs.dz_m <= 0.0:
        raise ValueError("dx_m and dz_m must be > 0.")
    if not (0.0 < inputs.target_reflection < 1.0):
        raise ValueError("target_reflection must be in (0, 1).")
    if inputs.kappa_max < 1.0:
        raise ValueError("kappa_max must be >= 1.")
    if inputs.alpha_fraction_of_omega0 < 0.0:
        raise ValueError("alpha_fraction_of_omega0 must be >= 0.")
    if inputs.f0_hz <= 0.0:
        raise ValueError("f0_hz must be > 0.")
    if inputs.vmax_m_per_s <= 0.0:
        raise ValueError("vmax_m_per_s must be > 0.")

    thickness_m = float(inputs.lpml) * min(float(inputs.dx_m), float(inputs.dz_m))
    m = max(1, int(inputs.poly_order_m))
    sigma_max = -((m + 1.0) * float(inputs.vmax_m_per_s) * log(float(inputs.target_reflection))) / (2.0 * thickness_m)
    alpha_max = float(inputs.alpha_fraction_of_omega0) * (2.0 * pi * float(inputs.f0_hz))
    return PmlHeuristicOutputs(
        pml_amax=alpha_max,
        pml_kmax=float(inputs.kappa_max),
        pml_smax=sigma_max,
        thickness_m=thickness_m,
    )


def recommend_workshop_pml_parameters(
    *,
    lpml: int,
    dx_m: float,
    dz_m: float,
    f0_hz: float,
    epsr: float = 7.0,
    sigma_mid_s_per_m: float = 0.255,
    target_reflection: float = 1e-6,
    kappa_max: float = 11.0,
    alpha_fraction_of_omega0: float = 1.0,
    poly_order_m: int = 3,
):
    vmax = lossy_medium_phase_velocity(
        f_hz=f0_hz,
        mu_h_per_m=MU0,
        eps_f_per_m=EPS0 * float(epsr),
        sigma_s_per_m=float(sigma_mid_s_per_m),
    )
    outputs = recommend_pml_parameters(
        PmlHeuristicInputs(
            lpml=int(lpml),
            dx_m=float(dx_m),
            dz_m=float(dz_m),
            f0_hz=float(f0_hz),
            vmax_m_per_s=float(vmax),
            target_reflection=float(target_reflection),
            kappa_max=float(kappa_max),
            alpha_fraction_of_omega0=float(alpha_fraction_of_omega0),
            poly_order_m=int(poly_order_m),
        )
    )
    return outputs, float(vmax)


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
    "DesignInputs",
    "DesignOutputs",
    "PmlHeuristicInputs",
    "PmlHeuristicOutputs",
    "enforce_rss_min_value",
    "interpolate_rss_python",
    "lossy_medium_phase_velocity",
    "recommend_pml_parameters",
    "recommend_design",
    "recommend_workshop_pml_parameters",
    "update_modcfg_for_workshop",
]
