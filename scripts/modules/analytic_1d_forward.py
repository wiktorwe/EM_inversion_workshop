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
`rockem.greens.greens_layered_2d`) gives
the EXACT 2D line-source answer for a 1D layered earth directly, with no
correction constants: it is the analytic counterpart of a `WavesEmTE2D`
`source_field="HX"` run, validated against explicit TE2D FDTD to within a
few percent (see that repo's README).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.modules.rockem_bridge import (
    GreensSolverError,
    magnetic_line_source_fields_layered,
    magnetic_z_line_source_fields_layered,
)

# Analytic line-source solver per TE2D source component. Both return
# ``(Ey, Hx, Hz)`` per unit source and share signature, cost and guarantees.
# HX = Kx (source_type=3), HZ = Kz (source_type=5).
_SOURCE_SOLVERS = {
    "HX": magnetic_line_source_fields_layered,
    "HZ": magnetic_z_line_source_fields_layered,
}

_CONTRASTED_INTERFACE_MSG = "contrasted layer interface"

# --- kx-quadrature policy ----------------------------------------------------
#
# The solver integrates over horizontal wavenumber with a fixed Gauss-Legendre
# rule of `n_nodes` points on `[0, lam_max]`, and `_default_lam_max` sizes
# `lam_max` as `40 / delta_min` (the model's MOST conductive layer). That
# default is validated upstream on rockem-suite's own test stacks, but it is a
# TRUNCATION limit, and truncation converges cleanly and WRONGLY in `n_nodes` -
# more nodes only refine a range that was already too short.
#
# Measured on THIS workshop's 1D inversion prior (5 layers, 2-100 Ohm-m,
# 5-50 m thicknesses, the survey's own 1/2/4/6 kHz and +-13.1/25.3 m offsets),
# against an 8x-lam_max reference - see `scripts/experiments/kx_convergence.py`:
#
#     n_nodes  lam_max x   worst error
#         120          1      1.69 %      <- shipped upstream default
#         120          2      0.22 %      <- ADOPTED
#         240          2      0.22 %      <- doubling n_nodes alone: NO change
#         240          4      0.004 %
#         480          8      failed (LinAlgError: singular layer matrix)
#
# 1.69 % is over half the workshop's 3 % relative-error floor, so it is not
# negligible for the inversion. Doubling `n_nodes` at the same `lam_max` changes
# NOTHING (0.22 % either way) - conclusive evidence that the residual is
# truncation, not resolution.
#
# WHY 2 AND NOT 4, even though 4 is 50x more accurate. Accuracy is worthless on
# a model the solver cannot evaluate. rockem-suite's gotchas warn that this
# solver has a silent-overflow failure mode and must be guarded inside any
# automated search; pushing `lam_max` up makes that failure MUCH more likely,
# because the layer matrix carries exp(+-gamma*thickness) with
# gamma ~ lam_max. Measured over 400 draws from this workshop's own 1D prior
# (5 layers, 1-150 Ohm-m, 5-50 m thicknesses):
#
#     multiplier   evaluated   rejected (guarded)   LinAlgError (hard fail)
#          1           392            8                     0
#          2           392            8                     0
#          3           376           13                    11
#          4           301           34                    65   <- 16 % of the prior
#
# At 4 a QUARTER of the prior becomes unevaluable, which would cripple a
# differential-evolution search far more than a 0.2 % forward error ever could.
# At 2 the robustness is identical to the shipped default while 87 % of the
# truncation error is removed. That is the trade actually worth taking.
#
# `LAM_MAX_MULTIPLIER` is applied to `_default_lam_max`, with `n_nodes` scaled by
# the same factor so node DENSITY is unchanged.
#
# The 15-layer lateral-average calibration Earth is ALREADY converged at
# multiplier 1 (measured 0.000 % change at every frequency), so this does not
# move the fitted C(f) - it only helps the inversion's thin, high-contrast
# candidate models.
LAM_MAX_MULTIPLIER = 2.0
_ANNOUNCED = False


def announce_quadrature_policy(force: bool = False) -> str:
    """Print the kx-quadrature policy once per process.

    A numeric switch that never says what it is set to is how a change stays
    silently inert (or silently active) for a whole test cycle. Every entry
    point that evaluates the analytic forward calls this.
    """
    global _ANNOUNCED
    msg = (f"[analytic_1d_forward] kx quadrature: lam_max = {LAM_MAX_MULTIPLIER:g} x "
           f"_default_lam_max, n_nodes scaled by the same factor "
           f"(node density unchanged)")
    if force or not _ANNOUNCED:
        print(msg)
        _ANNOUNCED = True
    return msg


def resolve_quadrature(layers: List[Layer1D], freq_hz: float, n_nodes: int,
                       lam_max: Optional[float]) -> Tuple[int, Optional[float]]:
    """(n_nodes, lam_max) actually handed to the solver for one frequency.

    An explicit `lam_max` is passed through untouched (convergence studies need
    to set it exactly); `None` means "apply the policy above".
    """
    if lam_max is not None:
        return int(n_nodes), float(lam_max)
    from rockem.greens.greens_layered_2d import _default_lam_max
    mult = float(LAM_MAX_MULTIPLIER)
    return int(round(n_nodes * mult)), mult * _default_lam_max(layers, float(freq_hz))


class ForwardRejected(Exception):
    """Raised by `forward_1d_gains` when the underlying analytic solver
    cannot evaluate a candidate model (see `GreensSolverError`) or returns a
    non-finite result. Callers driving an automated search (differential
    evolution, dual annealing) should catch this and assign an `inf` cost -
    never let it propagate as a NaN into a misfit."""


@dataclass
class Layer1D:
    """Duck-types `rockem.model.LayerSpec` (the fields `layers_to_stack`/
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


def _is_contrasted_interface_rejection(exc: GreensSolverError) -> bool:
    return _CONTRASTED_INTERFACE_MSG in str(exc)


def forward_1d_gains(
    rho: np.ndarray,
    thickness: np.ndarray,
    freqs_hz: np.ndarray,
    off_x: np.ndarray,
    tx_depth_m: float,
    rx_depth_m: float | np.ndarray,
    eps_r: float,
    n_nodes: int = 120,
    lam_max: Optional[float] = None,
    source_field: str = "HX",
    allow_empymod_fallback: bool = False,
    stats: Optional[Dict[str, bool]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Complex (Hx, Hz) channel gain per unit source, shape [nfreq, nrx].

    ``source_field`` selects which magnetic line source is modelled: ``"HX"``
    (Kx, the TE2D engine's ``source_type=3``) or ``"HZ"`` (Kz,
    ``source_type=5``). It MUST match the source that produced the data being
    fitted. This used to be hardcoded to Kx, so feeding it Kz data modelled the
    wrong source silently - the same class of error as an inversion cfg whose
    ``source_type`` does not match its shot gather, but with no cfg to inspect.

    `rx_depth_m` may be a SCALAR (all receivers at one depth) or a PER-RECEIVER
    array matching `off_x`, exactly like `fdtd_analytic_calibration`'s
    `layered_analytic_gains`/`homogeneous_analytic_gains`. Receivers are grouped
    by unique depth and one solve is issued per (frequency, depth), so a
    depth-offset receiver array costs no more than it has to. Passing a scalar
    where the receivers actually sit at different depths silently models them
    all at one depth - harmless for this workshop's default colinear,
    zero-depth-offset survey, wrong the moment anyone uses depth-offset
    receivers.

    `lam_max` overrides the solver's kx-truncation limit (default `None` =
    `_default_lam_max`, sized off the model's MOST conductive layer). It is
    exposed only so `check_kx_convergence` can vary it - production callers
    should leave it `None`.

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

    SAME-DEPTH RECEIVERS ARE FINE. The solver splits the primary (closed
    form, in the source layer's own whole space) from the secondary
    (quadrature), so `rx_depth_m == tx_depth_m` - this workshop's own
    default `gz0 == sz0` - evaluates correctly. The old "needs >= ~0.15
    skin depths of depth offset" restriction applied to the pre-
    decomposition solver and no longer exists.

    What DOES still raise `ForwardRejected`, all via `GreensSolverError`:
    - a receiver at the source point (`offset ~ 0` AND same depth) - the
      true line-source field is genuinely singular there;
    - the SOURCE within `0.01 * delta_min` of a CONTRASTED interface. The
      solver centres the layer stack on `tx_depth_m`, so a candidate whose
      cumulative thicknesses put an interface at the middle of the span
      lands on the transmitter. This is a thin forbidden slab, not a
      systematic failure: measured ~0.6% of draws over this workshop's
      default 5-layer prior (99 m span, 2-6 kHz, 2-100 Ohm-m). Rejected
      candidates simply cost `inf`;
    - a receiver in a DIFFERENT layer than the source but within
      `0.3 * delta_min` of it in depth. Cannot fire for a colinear survey
      (`dz == 0` implies same layer).

    When `allow_empymod_fallback=True`, contrasted-interface rejections fall
    back to the slower empymod y-integrated line-source reference (requires
    `pip install empymod`). Other rejections still raise `ForwardRejected`.
    If `stats` is provided, it receives `stats["empymod_fallback"]=True` when
    any frequency used the fallback.
    """
    source_field = str(source_field).upper()
    if source_field not in _SOURCE_SOLVERS:
        raise ValueError(
            f"source_field must be one of {sorted(_SOURCE_SOLVERS)}, got {source_field!r}"
        )
    solver = _SOURCE_SOLVERS[source_field]

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

    rx_depths = np.asarray(rx_depth_m, dtype=float).reshape(-1)
    if rx_depths.size == 1:
        rx_depths = np.full(off_x.shape, float(rx_depths[0]), dtype=float)
    if rx_depths.shape != off_x.shape:
        raise ValueError(
            f"rx_depth_m size {rx_depths.size} does not match off_x size {off_x.size}"
        )

    nfreq, nrx = freqs_hz.size, off_x.size
    hx = np.full((nfreq, nrx), np.nan, dtype=complex)
    hz = np.full((nfreq, nrx), np.nan, dtype=complex)
    used_empymod_fallback = False
    for ifreq, f in enumerate(freqs_hz):
        n_eff, lam_eff = resolve_quadrature(layers, float(f), n_nodes, lam_max)
        for depth in np.unique(rx_depths):
            mask = rx_depths == depth
            try:
                _, hx_f, hz_f = solver(
                    off_x[mask], float(f), layers, float(tx_depth_m), rx_depth_m=float(depth),
                    n_nodes=n_eff, lam_max=lam_eff,
                )
            except np.linalg.LinAlgError as exc:
                # The layer matrix went singular - rockem-suite's documented
                # overflow mode for this solver, made more likely by a larger
                # lam_max. It is a hard failure, not a NaN, so it would escape
                # an automated search as a crash rather than a rejected
                # candidate. Convert it to a rejection like every other
                # unevaluable model.
                raise ForwardRejected(
                    f"layer matrix singular at f={float(f):g} Hz "
                    f"(lam_max x{LAM_MAX_MULTIPLIER:g}): {exc}"
                ) from exc
            except GreensSolverError as exc:
                if allow_empymod_fallback and _is_contrasted_interface_rejection(exc):
                    try:
                        from scripts.modules.empymod_line_source import forward_empymod_line_gains
                    except ImportError as import_exc:
                        raise ForwardRejected(
                            f"{exc}; empymod line-source fallback unavailable ({import_exc})"
                        ) from import_exc
                    try:
                        hx_fb, hz_fb = forward_empymod_line_gains(
                            rho, thickness, np.asarray([f]), off_x[mask], tx_depth_m,
                            float(depth), eps_r,
                        )
                    except Exception as fb_exc:
                        raise ForwardRejected(
                            f"{exc}; empymod line-source fallback failed ({fb_exc})"
                        ) from fb_exc
                    hx_f, hz_f = hx_fb[0], hz_fb[0]
                    used_empymod_fallback = True
                    warnings.warn(
                        "analytic forward rejected on contrasted interface; "
                        "used empymod line-source fallback",
                        stacklevel=2,
                    )
                else:
                    raise ForwardRejected(str(exc)) from exc
            hx[ifreq, mask] = hx_f
            hz[ifreq, mask] = hz_f

    if not (np.all(np.isfinite(hx)) and np.all(np.isfinite(hz))):
        raise ForwardRejected("non-finite analytic forward result")
    if stats is not None and used_empymod_fallback:
        stats["empymod_fallback"] = True
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
    rel_tol: float = 3e-3,
    seed: int = 0,
) -> dict:
    """Spot-check the solver's kx-grid sizing across the ACTUAL prior bounds.

    The kx integral is a fixed Gauss-Legendre rule of `n_nodes` points on
    `[0, lam_max]`, and it has TWO independent failure modes:

    * **resolution** - too few nodes for the chosen range. Doubling `n_nodes`
      reveals it.
    * **truncation** - `lam_max` too short to reach the model's finest feature
      (the MOST conductive layer's skin depth). This converges CLEANLY and
      WRONGLY in `n_nodes`, because extra nodes only refine a range that was
      already too short. rockem-suite's gotchas record up to ~15 % error of
      exactly this kind on a 13x-contrast stack, and its own self-check calls
      lam_max-doubling "the decisive signal - n_nodes-doubling alone is blind".

    This function therefore runs BOTH legs. An earlier version doubled only
    `n_nodes`, which made it structurally incapable of detecting the very error
    it was written to guard against.

    Both Hx and Hz are compared, each normalised by its OWN row scale rather
    than pointwise: Hz is odd in offset and passes through zero, so a pointwise
    relative metric explodes at a near-null while the absolute change is
    negligible. A component whose scale is below `1e-6` of the co-component is
    a genuine null (exactly zero for a homogeneous draw) and is skipped.

    Run this once before trusting an inversion over a new or wider prior.

    ON `rel_tol`, because the default changed and a wrong threshold makes this
    check worse than useless in EITHER direction. The inherited default was
    1e-4, which suited the old n_nodes-only test - that leg returns ~1e-10, so
    it passed unconditionally and told you nothing. The lam_max leg legitimately
    sits near 1e-3 (measured 1.4e-3 over this workshop's own prior at the
    shipped quadrature policy), so carrying 1e-4 over to it made the check FAIL
    unconditionally instead, which is equally uninformative.

    The threshold is now anchored to the workshop's own uncertainty floor:
    `fdtd_analytic_calibration.VALIDATED_REL_ERROR_FLOOR` is 3 % of |FDTD|, and
    a forward-model error is harmless when it is well inside that. `rel_tol`
    defaults to a TENTH of the floor (0.3 %), so the shipped configuration
    reports converged with real margin, while a prior wide enough to matter
    still trips it. The verdict is reported at three levels rather than two, so
    "inside the noise floor but not negligible" is visible instead of being
    rounded to pass or fail.
    """
    from scripts.modules.rockem_bridge import magnetic_line_source_fields_layered as _solver
    from rockem.greens.greens_layered_2d import _default_lam_max

    rng = np.random.default_rng(seed)
    worst = {"n_nodes": 0.0, "lam_max": 0.0}
    n_rejected = 0
    per_draw = []

    def _row_change(a, b, reference):
        scale = float(np.max(np.abs(a)))
        if scale <= 1e-6 * max(float(np.max(np.abs(reference))), 1e-300):
            return 0.0
        return float(np.max(np.abs(a - b)) / max(scale, 1e-300))

    for _ in range(n_draws):
        rho = np.exp(rng.uniform(np.log(rho_bounds[0]), np.log(rho_bounds[1]), size=n_layers))
        thickness = np.exp(rng.uniform(np.log(thickness_bounds[0]), np.log(thickness_bounds[1]), size=max(0, n_layers - 1)))
        try:
            layers = layers_from_rho_thk(rho, thickness, eps_r)
            lam0 = min(_default_lam_max(layers, float(f)) for f in np.asarray(freqs_hz, dtype=float).reshape(-1))
            hx1, hz1 = forward_1d_gains(rho, thickness, freqs_hz, off_x, tx_depth_m, rx_depth_m,
                                        eps_r, n_nodes=n_nodes_default)
            hx2, hz2 = forward_1d_gains(rho, thickness, freqs_hz, off_x, tx_depth_m, rx_depth_m,
                                        eps_r, n_nodes=2 * n_nodes_default)
            hx3, hz3 = forward_1d_gains(rho, thickness, freqs_hz, off_x, tx_depth_m, rx_depth_m,
                                        eps_r, n_nodes=4 * n_nodes_default, lam_max=4.0 * lam0)
        except ForwardRejected:
            n_rejected += 1
            continue
        rel_n = max(_row_change(hx1, hx2, hx1), _row_change(hz1, hz2, hx1))
        rel_l = max(_row_change(hx1, hx3, hx1), _row_change(hz1, hz3, hx1))
        per_draw.append({"n_nodes": float(rel_n), "lam_max": float(rel_l)})
        worst["n_nodes"] = max(worst["n_nodes"], rel_n)
        worst["lam_max"] = max(worst["lam_max"], rel_l)

    decisive = worst["lam_max"]
    worst_any = max(worst.values())
    converged = worst_any < rel_tol
    noise_floor = 0.03          # VALIDATED_REL_ERROR_FLOOR, imported lazily below
    try:
        from scripts.modules.fdtd_analytic_calibration import VALIDATED_REL_ERROR_FLOOR
        noise_floor = float(VALIDATED_REL_ERROR_FLOOR)
    except Exception:
        pass
    if converged:
        verdict = "converged"
    elif worst_any < noise_floor:
        verdict = "inside the noise floor"
    else:
        verdict = "NOT converged"
    return {
        "verdict": verdict,
        "noise_floor": noise_floor,
        "n_draws": n_draws, "n_rejected": n_rejected,
        "worst_relative_change": max(worst.values()),
        "worst_relative_change_n_nodes": worst["n_nodes"],
        "worst_relative_change_lam_max": decisive,
        "per_draw_relative_change": per_draw, "converged": converged, "rel_tol": rel_tol,
        "notes": f"{verdict} (tol {rel_tol:.1e}, noise floor {noise_floor:.0%})"
                 + f"; n_nodes leg {worst['n_nodes']:.2e}, lam_max leg {decisive:.2e}"
                 + " - the lam_max leg is the decisive one, n_nodes alone is blind to truncation"
                 + ("" if converged else
                    ("; well inside the uncertainty floor, so it does not limit the inversion"
                     if worst_any < noise_floor else
                     " - raise lam_max (truncation) and/or n_nodes (resolution)"))
                 + (f"; {n_rejected}/{n_draws} draws rejected by the solver" if n_rejected else ""),
    }


__all__ = [
    "LAM_MAX_MULTIPLIER",
    "ForwardRejected",
    "Layer1D",
    "announce_quadrature_policy",
    "resolve_quadrature",
    "layers_from_rho_thk",
    "forward_1d_gains",
    "check_kx_convergence",
]
