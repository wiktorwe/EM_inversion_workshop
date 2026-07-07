"""1D layered forward model for the workshop's magnetic (Hx) line source,
using rockem-suite's validated `magnetic_line_source_fields_layered` -
replaces `empymod_1d_forward.py`'s `empymod.dipole` point-dipole forward
entirely.

Why this replaces empymod here: the workshop's 2D FDTD data comes from a
`Jy`/`Kx`-invariant LINE source (`WavesEmTE2D`), not a 3D point dipole -
empymod has no native line-source mode. The old code got a "1D reference"
by calling `empymod.dipole` directly and then papering over the resulting
2D-vs-3D mismatch with ad-hoc corrections (a flat -180 degree phase
constant, a sqrt(offset) amplitude "spreading" factor, a forced
time-derivative on the FDTD side) - see the git history this workshop
inherited. `magnetic_line_source_fields_layered` (rockem-suite,
`doc/examples/validate_layered_1d_model/shared/greens_layered_2d.py`) gives
the EXACT 2D line-source answer for a 1D layered earth directly, with no
correction constants: it is the analytic counterpart of a `WavesEmTE2D`
`source_field="HX"` run, validated against explicit TE2D FDTD to within a
few percent (see that repo's README).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from scripts.modules.rockem_bridge import GreensSolverError, magnetic_line_source_fields_layered


class ForwardRejected(Exception):
    """Raised by `forward_1d_gains` when the underlying analytic solver
    cannot evaluate a candidate model (see `GreensSolverError`) or returns a
    non-finite result. Callers driving an automated search (differential
    evolution, dual annealing) should catch this and assign an `inf` cost -
    never let it propagate as a NaN into a misfit."""


@dataclass
class Layer1D:
    """Duck-types `rockem.model.LayerSpec` (the fields `layers_to_empymod`/
    `magnetic_line_source_fields_layered` actually read: `resistivity_ohm_m`,
    `thickness_m`, `permittivity`) - the last layer's `thickness_m` MUST be
    `None` (halfspace)."""
    resistivity_ohm_m: float
    thickness_m: Optional[float]
    permittivity: float


def layers_from_rho_thk(rho: np.ndarray, thickness: np.ndarray, eps_r: float) -> List[Layer1D]:
    """Build a `Layer1D` stack from inversion-parameterization arrays.

    `rho` has length n_layers; `thickness` has length n_layers-1 (finite
    layers only - the last layer is always the halfspace). `eps_r` is
    shared by every layer (see `forward_1d_gains` for why it should be the
    FD run's own `eps_r_used`, not a separately-guessed physical value).
    """
    rho = np.asarray(rho, dtype=float).reshape(-1)
    thickness = np.asarray(thickness, dtype=float).reshape(-1)
    n_layers = int(rho.size)
    if n_layers < 1:
        raise ValueError("rho must contain at least one layer.")
    if thickness.size != max(0, n_layers - 1):
        raise ValueError(f"thickness length mismatch: expected {n_layers - 1}, got {thickness.size}")
    layers = [Layer1D(float(rho[i]), float(thickness[i]), float(eps_r)) for i in range(n_layers - 1)]
    layers.append(Layer1D(float(rho[-1]), None, float(eps_r)))
    return layers


def forward_1d_gains(
    rho: np.ndarray,
    thickness: np.ndarray,
    freqs_hz: np.ndarray,
    off_x: np.ndarray,
    tx_depth_m: float,
    rx_depth_m: float,
    eps_r: float,
    n_nodes: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """Complex (Hx, Hz) channel gain per unit Kx source, shape [nfreq, nrx].

    `off_x` are SIGNED tx-relative offsets (matches the survey convention
    in `01_fw_setup` - `gx0`/`dgx` are already tx-relative); the sign
    matters because Hz is odd in offset (see `magnetic_line_source_fields_
    layered`'s docstring on component symmetry) - do not pass `abs(off_x)`.
    `eps_r` should be the FORWARD RUN's own `eps_r_used` (from
    `design_explicit_fd` / `setup_metadata.json`), not a separately-guessed
    physical value - see module docstring; this keeps the analytic
    reference and the FDTD "truth" data on the exact same numerical
    footing (same displacement-current assumption), so the FDTD-vs-analytic
    residual reflects only genuine discretization error, not an eps_r
    mismatch on top of it.

    Raises `ForwardRejected` (never returns NaN) if the analytic solver
    cannot evaluate this candidate model - callers in an automated search
    must catch this and assign an `inf` cost.

    IMPORTANT: also raises `ForwardRejected` if `rx_depth_m` is too close
    to `tx_depth_m` (within ~0.15 skin depths) - the underlying kx-integral
    truncation is confirmed WRONG (not just unstable), not merely
    under-resolved, that close to the source (see the rockem-suite skill's
    gotcha on this). A survey with receivers at the SAME depth as the
    source (a common logging/borehole convention, and this workshop's own
    default `gz0 == sz0`) will hit this on every call - the 1D inversion
    stage needs a genuine depth offset between source and receivers to use
    this solver as a reference/calibration tool.
    """
    rho = np.asarray(rho, dtype=float).reshape(-1)
    thickness = np.asarray(thickness, dtype=float).reshape(-1)
    freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
    off_x = np.asarray(off_x, dtype=float).reshape(-1)

    if np.any(rho <= 0.0):
        raise ForwardRejected("non-positive resistivity in candidate model")
    if thickness.size and np.any(thickness <= 0.0):
        raise ForwardRejected("non-positive thickness in candidate model")

    try:
        layers = layers_from_rho_thk(rho, thickness, eps_r)
    except ValueError as exc:
        raise ForwardRejected(str(exc)) from exc

    nfreq, nrx = freqs_hz.size, off_x.size
    hx = np.full((nfreq, nrx), np.nan, dtype=complex)
    hz = np.full((nfreq, nrx), np.nan, dtype=complex)
    for ifreq, f in enumerate(freqs_hz):
        try:
            _, hx_f, hz_f = magnetic_line_source_fields_layered(
                off_x, float(f), layers, float(tx_depth_m), rx_depth_m=float(rx_depth_m), n_nodes=n_nodes,
            )
        except GreensSolverError as exc:
            raise ForwardRejected(str(exc)) from exc
        hx[ifreq, :] = hx_f
        hz[ifreq, :] = hz_f

    if not (np.all(np.isfinite(hx)) and np.all(np.isfinite(hz))):
        raise ForwardRejected("non-finite analytic forward result")
    return hx, hz


def check_kx_convergence(
    rho_bounds: Tuple[float, float],
    thickness_bounds: Tuple[float, float],
    n_layers: int,
    freqs_hz: Sequence[float],
    off_x: Sequence[float],
    tx_depth_m: float,
    rx_depth_m: float,
    eps_r: float,
    n_draws: int = 25,
    n_nodes_default: int = 120,
    rel_tol: float = 1e-4,
    seed: int = 0,
) -> dict:
    """Spot-check `_default_lam_max`'s kx-grid sizing heuristic across the
    ACTUAL prior bounds used by this inversion, not just the one model
    `greens_layered_2d.py` ships validated against - see the rockem-suite
    skill's gotcha on this (`_default_lam_max` sizes off the model's most
    conductive layer; wrong-but-stable-looking convergence is possible if
    the heuristic under-resolves a high-contrast draw). Doubles `n_nodes`
    at `n_draws` random points spanning the prior and reports the worst
    relative Hx change - run this once before trusting an inversion over a
    new/wider prior.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    n_rejected = 0
    per_draw = []
    for _ in range(n_draws):
        rho = np.exp(rng.uniform(np.log(rho_bounds[0]), np.log(rho_bounds[1]), size=n_layers))
        thickness = np.exp(rng.uniform(np.log(thickness_bounds[0]), np.log(thickness_bounds[1]), size=max(0, n_layers - 1)))
        try:
            hx1, _ = forward_1d_gains(rho, thickness, freqs_hz, off_x, tx_depth_m, rx_depth_m, eps_r, n_nodes=n_nodes_default)
            hx2, _ = forward_1d_gains(rho, thickness, freqs_hz, off_x, tx_depth_m, rx_depth_m, eps_r, n_nodes=2 * n_nodes_default)
        except ForwardRejected:
            n_rejected += 1
            continue
        rel = np.max(np.abs(hx2 - hx1) / np.maximum(np.abs(hx1), 1e-300))
        per_draw.append(float(rel))
        worst = max(worst, float(rel))
    converged = worst < rel_tol
    return {
        "n_draws": n_draws, "n_rejected": n_rejected, "worst_relative_change": worst,
        "per_draw_relative_change": per_draw, "converged": converged, "rel_tol": rel_tol,
        "notes": ("converged" if converged else "NOT converged - raise n_nodes_default or shrink the prior")
                 + (f"; {n_rejected}/{n_draws} draws rejected by the solver" if n_rejected else ""),
    }


__all__ = [
    "ForwardRejected",
    "Layer1D",
    "layers_from_rho_thk",
    "forward_1d_gains",
    "check_kx_convergence",
]
