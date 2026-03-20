"""Plotting helpers for layered block1D models and uncertainty bars."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from .model import split_model


def layer_depths(thickness: np.ndarray) -> np.ndarray:
    """Return interface depths from thickness vector."""
    return np.cumsum(np.asarray(thickness, dtype=float))


def plot_block_model(
    model: np.ndarray,
    n_layers: int,
    *,
    ax: plt.Axes | None = None,
    label: str = "model",
    color: str = "C0",
    xscale: str = "log",
) -> plt.Axes:
    """Plot layered resistivity model as a stair-step curve."""
    thk, rho = split_model(np.asarray(model, dtype=float), n_layers=n_layers)
    z = layer_depths(thk)
    # Create step coordinates: each resistivity repeated across its layer.
    z_plot = np.hstack([[0.0], np.repeat(z, 2), [z[-1] * 1.2]])
    rho_plot = np.repeat(rho, 2)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(rho_plot, z_plot, color=color, label=label)
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Resistivity")
    ax.set_ylim(max(z_plot), 0.0)
    if xscale == "log":
        ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_model_with_errorbars(
    model: np.ndarray,
    sigma_abs: np.ndarray,
    n_layers: int,
    *,
    ax: plt.Axes | None = None,
    label: str = "model",
    color: str = "C1",
) -> plt.Axes:
    """Plot block model with tutorial-like x/y error bars.

    - Resistivity uncertainty shown as horizontal bars at layer midpoints.
    - Thickness uncertainty shown as vertical bars at interfaces.
    """
    model = np.asarray(model, dtype=float)
    sigma_abs = np.asarray(sigma_abs, dtype=float)
    if model.size != sigma_abs.size:
        raise ValueError("model and sigma_abs size mismatch.")

    n_thk = n_layers - 1
    thk = model[:n_thk]
    rho = model[n_thk:]
    sthk = sigma_abs[:n_thk]
    srho = sigma_abs[n_thk:]

    z = layer_depths(thk)
    mids = np.hstack([z - thk / 2.0, z[-1] * 1.1])
    rho_mean = np.sqrt(rho[:-1] * rho[1:])

    ax = plot_block_model(model, n_layers=n_layers, ax=ax, label=label, color=color)
    ax.errorbar(rho, mids, marker="*", ls="None", xerr=srho, color="k")
    ax.errorbar(rho_mean, z, marker="*", ls="None", yerr=sthk, color="k")
    ax.legend()
    return ax

