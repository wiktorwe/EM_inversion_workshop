"""Example integration of pygimli.empy_blockinv with a user forward model.

This script is intentionally verbose and heavily commented to make adaptation
to Empymod straightforward.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the inversion core without importing the plotting submodule at import-time.
# This keeps the example runnable in environments that may not have matplotlib.
from third_party.empy_blockinv.interfaces import InversionConfig
from third_party.empy_blockinv.model import pack_model
from third_party.empy_blockinv.inversion import run_block1d_inversion
from third_party.empy_blockinv.covariance import uncertainty_from_result

from scripts.modules.empymod_1d_forward import forward_empymod_1d_layered_amp_phase


def wrap_phase_rad(phi: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi]."""
    phi = np.asarray(phi, dtype=float)
    return np.arctan2(np.sin(phi), np.cos(phi))


def empymod_forward_phase(model: np.ndarray, ctx: dict) -> np.ndarray:
    """Empymod-based forward producing concatenated phase predictions.

    Returns a 1D prediction vector of length `2 * nfreq * nrx` in the fixed
    order:
      - Hx-Hx phases, flattened as [nfreq, nrx] then raveled
      - Hx-Hz phases, flattened as [nfreq, nrx] then raveled
    """
    n_layers = int(ctx["n_layers"])
    n_thk = n_layers - 1
    thk = np.asarray(model[:n_thk], dtype=float)
    rho = np.asarray(model[n_thk:], dtype=float)

    # Constrain total thickness ("span") so the inversion cannot cheat by
    # shrinking/expanding the depth window. The workshop notebooks do this
    # by parameterizing thicknesses in a fixed depth span.
    span_target = float(ctx["span_target"])
    thk_sum = float(np.sum(thk))
    if not np.isfinite(thk_sum) or thk_sum <= 0.0:
        thk_sum = 1e-12
    thk = thk / thk_sum * span_target

    freqs = np.asarray(ctx["freqs"], dtype=float)
    off_x = np.asarray(ctx["off_x"], dtype=float)
    off_z = np.asarray(ctx["off_z"], dtype=float)
    tilt_deg = float(ctx.get("tilt_deg", 0.0))
    ab_hxh = int(ctx.get("ab_hxh", 44))
    ab_hxhz = int(ctx.get("ab_hxhz", 46))

    # Forward helper returns amp/phase for both components.
    _amp_hxh, phi_hxh, _amp_hxhz, phi_hxhz = forward_empymod_1d_layered_amp_phase(
        rho=rho,
        thickness=thk,
        freqs=freqs,
        off_x=off_x,
        off_z=off_z,
        tilt_deg=tilt_deg,
        ab_hxh=ab_hxh,
        ab_hxhz=ab_hxhz,
    )

    # Guard against unexpected shapes so inversion failures are readable.
    phi_hxh = np.asarray(phi_hxh, dtype=float)
    phi_hxhz = np.asarray(phi_hxhz, dtype=float)
    if phi_hxh.ndim != 2 or phi_hxhz.ndim != 2:
        raise ValueError(f"Expected 2D phase arrays, got {phi_hxh.shape} and {phi_hxhz.shape}.")

    return np.concatenate([phi_hxh.ravel(), phi_hxhz.ravel()]).astype(float, copy=False)


def custom_phase_residual(
    observed: np.ndarray,
    predicted: np.ndarray,
    error: np.ndarray,
    ctx: dict,
) -> np.ndarray:
    """Optional custom misfit callback.

    You can implement phase wrapping logic here if needed.
    If not needed, a simple weighted residual works:
    r = (observed - predicted) / error
    """
    _ = ctx  # keep this to show context is available inside callback
    dphi = wrap_phase_rad(observed - predicted)
    residual = dphi / error

    return residual


def custom_cosine_residual(
    observed: np.ndarray,
    predicted: np.ndarray,
    error: np.ndarray,
    ctx: dict,
) -> np.ndarray:
    """Cosine-domain residual: compare cos(phase) instead of phase difference.

    This is typically more stable for wrapped phase because cos is continuous
    across the [-pi, pi] boundary.
    """
    _ = ctx
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    err = np.asarray(error, dtype=float)

    dcos = np.cos(obs) - np.cos(pred)

    # Approximate how phase error maps to cosine error:
    # d(cos(phi))/dphi = -sin(phi)  => |dcos| ~ |sin(phi)| * dphi.
    # Guard against sin(phi) ~ 0 where the linear approx would yield tiny
    # denominators (then enforce a reasonable minimum weight).
    min_cos_err = 0.25 * float(np.mean(err)) if np.isfinite(np.mean(err)) else 1e-6
    cos_err = np.clip(np.abs(np.sin(obs)) * err, min_cos_err, np.inf)
    return dcos / cos_err


def main() -> None:
    """Run a full inversion and plot uncertainty bars."""
    plot_enabled = True
    try:
        import matplotlib

        import matplotlib.pyplot as plt  # noqa: E402

        from third_party.empy_blockinv.plotting import plot_model_with_errorbars
        from third_party.empy_blockinv.plotting import plot_block_model
    except Exception:
        plot_enabled = False
        plt = None
        plot_model_with_errorbars = None
        plot_block_model = None

    n_layers = 4
    true_thk = np.array([5.0, 10.0, 20.0], dtype=float)
    true_rho = np.array([100.0, 20.0, 200.0, 50.0], dtype=float)
    true_model = pack_model(true_thk, true_rho)
    span_target = float(np.sum(true_thk))

    freqs = np.asarray([2000.0, 4000.0, 6000.0], dtype=float)
    # Compact geometry for a quick smoke test.
    off_x = np.linspace(10.0, 100.0, 8, dtype=float)
    off_z = np.zeros_like(off_x, dtype=float)
    tilt_deg = 0.0

    context = {
        "n_layers": n_layers,
        "freqs": freqs,
        "off_x": off_x,
        "off_z": off_z,
        "tilt_deg": tilt_deg,
        "ab_hxh": 44,
        "ab_hxhz": 46,
        "span_target": span_target,
    }

    observed_clean = empymod_forward_phase(true_model, context)
    rng = np.random.default_rng(1337)
    phase_error_abs = 0.03  # absolute phase error (rad)
    phase_error = np.full_like(observed_clean, phase_error_abs, dtype=float)
    observed_noisy = wrap_phase_rad(observed_clean + rng.normal(scale=phase_error_abs, size=observed_clean.shape))

    nfreq = int(freqs.size)
    nrx = int(off_x.size)
    expected_len = 2 * nfreq * nrx
    if observed_noisy.size != expected_len:
        raise AssertionError(f"Unexpected data vector length: got {observed_noisy.size}, expected {expected_len}.")

    cfg = InversionConfig(
        n_layers=n_layers,
        # If start_model is omitted, provide start_depth_min/start_depth_max
        # in the inversion call, and the package builds a log-spaced start.
        start_model=pack_model(
            thickness=np.array([4.0, 8.0, 16.0], dtype=float),
            resistivity=np.array([80.0, 80.0, 80.0, 80.0], dtype=float),
        ),
        lam=500.0,
        lambda_factor=0.8,
        max_iter=10,  # keep runtime reasonable for Empymod FD Jacobians
        min_dphi_percent=1.0,
        stop_at_chi1=False,
        callback_context=context,
        line_search=True,
    )

    result = run_block1d_inversion(
        observed=observed_noisy,
        error=phase_error,
        config=cfg,
        forward_fn=empymod_forward_phase,
        misfit_fn=custom_cosine_residual,
    )
    sigma_abs, sigma_rel = uncertainty_from_result(result)

    print("Final model:", result.model)
    print("Relative parameter uncertainty (log-domain approx):", sigma_rel)
    print("Final chi2:", result.chi2_history[-1])
    print(f"Synthetic/fit data size: {observed_noisy.size} (nfreq={nfreq}, nrx={nrx})")
    
    print("chi2_history:", result.chi2_history)
    print("model:", result.model)

    # Plot model with tutorial-like uncertainty bars.
    if plot_enabled and plot_model_with_errorbars is not None:
        # Rescale inverted thickness for plotting so true vs inverted share
        # the same physical depth window.
        n_thk = n_layers - 1
        inv_thk = np.asarray(result.model[:n_thk], dtype=float)
        inv_rho = np.asarray(result.model[n_thk:], dtype=float)
        inv_span = float(np.sum(inv_thk))
        k = span_target / inv_span if inv_span > 0.0 else 1.0
        inv_thk_plot = inv_thk * k
        model_for_plot = pack_model(inv_thk_plot, inv_rho)

        sigma_abs_plot = np.asarray(sigma_abs, dtype=float).copy()
        sigma_abs_plot[:n_thk] = sigma_abs_plot[:n_thk] * abs(k)

        ax = plot_model_with_errorbars(
            model=model_for_plot,
            sigma_abs=sigma_abs_plot,
            n_layers=n_layers,
            label="inverted",
            color="C1",
        )
        # Overlay the known true model for visual validation.
        if plot_block_model is not None:
            plot_block_model(
                model=true_model,
                n_layers=n_layers,
                ax=ax,
                label="true",
                color="C0",
            )
            ax.legend()
        ax.figure.suptitle("Block1D model with uncertainty bars")
        ax.figure.tight_layout()
        if plt is not None:
            plt.show()
    else:
        print("Plot skipped (matplotlib not available).")


if __name__ == "__main__":
    main()

