"""Matplotlib PDF figure writers for the workshop workflow report."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.modules.rss_model import conductivity_to_resistivity

_PHASE_YLIM = (-180.0, 180.0)


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _rho_limits(*grids: np.ndarray) -> tuple[float, float]:
    vals = []
    for g in grids:
        if g is None:
            continue
        a = np.asarray(g, dtype=float)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size:
            vals.append(a)
    if not vals:
        return 1.0, 100.0
    stacked = np.concatenate(vals)
    lo = float(np.nanpercentile(stacked, 2))
    hi = float(np.nanpercentile(stacked, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(stacked))
        hi = float(np.nanmax(stacked))
    if hi <= lo:
        hi = lo * 10.0 if lo > 0 else lo + 1.0
    return lo, hi


def _heatmap(ax, x, z, rho, *, vmin, vmax, cmap="jet", title="", cbar_label="Ohm-m"):
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    rho = np.asarray(rho, dtype=float)
    im = ax.pcolormesh(
        x, z, rho, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True
    )
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_aspect("auto")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    return im


def _mid_index(values: np.ndarray) -> int:
    vals = np.unique(np.asarray(values, dtype=int))
    if vals.size == 0:
        return 0
    return int(vals[vals.size // 2])


def survey_positions_from_meta(meta: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tx/Rx coordinates from ``setup_metadata.json`` survey fields."""
    ntx = int(meta.get("ntx") or 0)
    nrx = int(meta.get("nrx") or 0)
    tx0 = float(meta.get("tx0_m") or 0.0)
    tz0 = float(meta.get("tz0_m") or 0.0)
    dtx = float(meta.get("dtx_m") or 0.0)
    rx0 = float(meta.get("rx0_m") or 0.0)
    rz0 = float(meta.get("rz0_m") or 0.0)
    drx = float(meta.get("drx_m") or 0.0)
    tx_x = tx0 + dtx * np.arange(max(ntx, 0), dtype=float)
    tx_z = np.full(tx_x.shape, tz0, dtype=float)
    rx_x = rx0 + drx * np.arange(max(nrx, 0), dtype=float)
    rx_z = np.full(rx_x.shape, rz0, dtype=float)
    return tx_x, tx_z, rx_x, rx_z


def save_resistivity_survey_figure(
    x,
    z,
    rho,
    path: Path,
    *,
    tx_x=None,
    tx_z=None,
    rx_x=None,
    rx_z=None,
    title: str = "Resistivity model + survey geometry",
) -> Path:
    rho = np.asarray(rho, dtype=float)
    vmin, vmax = _rho_limits(rho)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    _heatmap(ax, x, z, rho, vmin=vmin, vmax=vmax, cmap="jet", title=title)
    if tx_x is not None and tx_z is not None and np.size(tx_x):
        ax.scatter(tx_x, tx_z, c="red", marker="x", s=36, linewidths=1.2, label="Sources", zorder=3)
    if rx_x is not None and rx_z is not None and np.size(rx_x):
        rec_step = max(1, int(np.ceil(np.size(rx_x) / 5000)))
        ax.scatter(
            np.asarray(rx_x)[::rec_step],
            np.asarray(rx_z)[::rec_step],
            c="white",
            s=8,
            edgecolors="k",
            linewidths=0.3,
            label="Receivers",
            zorder=2,
        )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", framealpha=0.8)
    return _save(fig, path)


def save_wavelet_figure(t, w, path: Path) -> Path:
    t = np.asarray(t, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    n = int(min(t.size, w.size))
    t, w = t[:n], w[:n]
    dt = float(t[1] - t[0]) if n > 1 else 1.0
    freqs = np.fft.rfftfreq(n, d=max(dt, 1e-30))
    amp = np.abs(np.fft.rfft(w))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    axes[0].plot(t, w, color="C0", lw=1.0)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Wavelet (time domain)")
    axes[1].plot(freqs, amp, color="C0", lw=1.0)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("|FFT|")
    axes[1].set_title("Wavelet amplitude spectrum")
    return _save(fig, path)


def _trace_subset(gains: Mapping, tx_id: int) -> tuple[np.ndarray, np.ndarray]:
    geo = gains["geometry"]
    tx = np.asarray(geo["tx_idx_per_trace"], dtype=int)
    rx_local = np.asarray(geo.get("rx_local_idx_per_trace", geo.get("rx_idx_per_trace")), dtype=int)
    idx = np.where(tx == int(tx_id))[0]
    if idx.size == 0:
        raise ValueError(f"No traces for transmitter {tx_id}")
    order = np.argsort(rx_local[idx])
    return idx[order], rx_local[idx][order]


def save_amp_phase_vs_rx_figure(gains: Mapping, path: Path, *, tx_id: Optional[int] = None) -> Path:
    """Hx/Hz amplitude and phase vs local rx, all frequencies overplotted."""
    geo = gains["geometry"]
    tx = np.asarray(geo["tx_idx_per_trace"], dtype=int)
    if tx_id is None:
        tx_id = _mid_index(tx)
    idx, rx = _trace_subset(gains, tx_id)
    freqs = np.asarray(gains.get("freqs", gains["Hx"].get("freqs", [])), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2), sharex=True)
    panels = (
        (axes[0, 0], "Hx", "amp_mean", "Hx amplitude", False),
        (axes[0, 1], "Hz", "amp_mean", "Hz amplitude", False),
        (axes[1, 0], "Hx", "phi_mean_rad", "Hx phase (deg)", True),
        (axes[1, 1], "Hz", "phi_mean_rad", "Hz phase (deg)", True),
    )
    for ax, comp, key, title, is_phase in panels:
        data = gains[comp]
        arr = np.asarray(data[key], dtype=float)
        for fi, f in enumerate(freqs):
            y = arr[fi, idx]
            if is_phase:
                y = np.rad2deg(y)
            ax.plot(rx, y, lw=1.0, marker="o", ms=2.5, label=f"{f:g} Hz")
        ax.set_title(title)
        ax.set_ylabel("Phase (deg)" if is_phase else "Amplitude")
        if is_phase:
            ax.set_ylim(*_PHASE_YLIM)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Local rx index")
    axes[1, 1].set_xlabel("Local rx index")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), fontsize=8, frameon=False)
        fig.subplots_adjust(top=0.88)
    fig.suptitle(f"Modelled channel-gain amp/phase vs rx (Tx {tx_id})", y=0.98)
    return _save(fig, path)


def save_calibration_c_figure(
    freqs,
    c_arr,
    path: Path,
    *,
    scatter_hx_pct=None,
    scatter_hz_pct=None,
    method: str = "",
) -> Path:
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    c_arr = np.asarray(c_arr, dtype=complex).reshape(-1)
    n = min(freqs.size, c_arr.size)
    freqs, c_arr = freqs[:n], c_arr[:n]
    has_scatter = scatter_hx_pct is not None or scatter_hz_pct is not None
    ncols = 3 if has_scatter else 2
    fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols + 1.2, 3.2))
    axes = np.atleast_1d(axes)
    axes[0].plot(freqs, np.abs(c_arr), "o-", lw=1.1)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("|C|")
    axes[0].set_title("|C(f)|")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(freqs, np.angle(c_arr, deg=True), "o-", lw=1.1, color="C1")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (deg)")
    axes[1].set_ylim(*_PHASE_YLIM)
    axes[1].set_title("arg C(f)")
    axes[1].grid(True, alpha=0.3)
    if has_scatter:
        if scatter_hx_pct is not None:
            axes[2].plot(freqs, np.asarray(scatter_hx_pct, dtype=float)[:n], "o-", label="Hx")
        if scatter_hz_pct is not None:
            axes[2].plot(freqs, np.asarray(scatter_hz_pct, dtype=float)[:n], "s-", label="Hz")
        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].set_ylabel("Relative scatter (%)")
        axes[2].set_title("Calibration scatter")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
    title = "FDTD–analytic calibration C(f)"
    if method:
        title += f" ({method})"
    fig.suptitle(title, y=1.02)
    return _save(fig, path)


def save_2d_model_compare_figure(
    x_true,
    z_true,
    rho_true,
    x_inv,
    z_inv,
    rho_inv,
    path: Path,
    *,
    inv_label: str = "Inverted",
) -> Path:
    vmin, vmax = _rho_limits(rho_true, rho_inv)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
    _heatmap(axes[0], x_true, z_true, rho_true, vmin=vmin, vmax=vmax, cmap="viridis", title="True model")
    _heatmap(axes[1], x_inv, z_inv, rho_inv, vmin=vmin, vmax=vmax, cmap="viridis", title=inv_label)
    fig.suptitle("2D resistivity: true vs inverted", y=1.02)
    return _save(fig, path)


def save_2d_slices_figure(
    x_true,
    z_true,
    rho_true,
    x_inv,
    z_inv,
    rho_inv,
    path: Path,
    *,
    x_pick: Optional[float] = None,
    z_pick: Optional[float] = None,
    inv_label: str = "Inverted",
) -> Path:
    x_true = np.asarray(x_true, dtype=float)
    z_true = np.asarray(z_true, dtype=float)
    x_inv = np.asarray(x_inv, dtype=float)
    z_inv = np.asarray(z_inv, dtype=float)
    if x_pick is None:
        x_pick = float(x_inv[x_inv.size // 2]) if x_inv.size else 0.0
    if z_pick is None:
        z_pick = float(z_inv[z_inv.size // 2]) if z_inv.size else 0.0

    def _col(x_arr, grid, pick):
        ix = int(np.argmin(np.abs(np.asarray(x_arr, dtype=float) - pick)))
        return np.asarray(grid, dtype=float)[:, ix]

    def _row(z_arr, grid, pick):
        iz = int(np.argmin(np.abs(np.asarray(z_arr, dtype=float) - pick)))
        return np.asarray(grid, dtype=float)[iz, :]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    axes[0].plot(_col(x_true, rho_true, x_pick), z_true, label="True", lw=1.2)
    axes[0].plot(_col(x_inv, rho_inv, x_pick), z_inv, label=inv_label, lw=1.2)
    axes[0].set_xlabel("Resistivity (Ohm-m)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()
    axes[0].set_title(f"vs depth at x = {x_pick:g} m")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(x_true, _row(z_true, rho_true, z_pick), label="True", lw=1.2)
    axes[1].plot(x_inv, _row(z_inv, rho_inv, z_pick), label=inv_label, lw=1.2)
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Resistivity (Ohm-m)")
    axes[1].set_title(f"vs x at z = {z_pick:g} m")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    return _save(fig, path)


def save_obs_vs_syn_figure(
    obs: Mapping,
    syn: Mapping,
    path: Path,
    *,
    tx_id: Optional[int] = None,
) -> Path:
    geo = obs["geometry"]
    tx = np.asarray(geo["tx_idx_per_trace"], dtype=int)
    if tx_id is None:
        tx_id = _mid_index(tx)
    idx, rx = _trace_subset(obs, tx_id)
    freqs = np.asarray(obs.get("freqs", obs["Hx"].get("freqs", [])), dtype=float)
    nfreq = int(freqs.size)
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2), sharex=True)
    specs = (
        (axes[0, 0], "Hx", "amp_mean", "Hx amplitude", False),
        (axes[0, 1], "Hz", "amp_mean", "Hz amplitude", False),
        (axes[1, 0], "Hx", "phi_mean_rad", "Hx phase (deg)", True),
        (axes[1, 1], "Hz", "phi_mean_rad", "Hz phase (deg)", True),
    )
    cmap = plt.cm.tab10
    for ax, comp, key, title, is_phase in specs:
        o = np.asarray(obs[comp][key], dtype=float)
        s = np.asarray(syn[comp][key], dtype=float)
        for fi, f in enumerate(freqs):
            color = cmap(fi % 10)
            yo = np.rad2deg(o[fi, idx]) if is_phase else o[fi, idx]
            ys = np.rad2deg(s[fi, idx]) if is_phase else s[fi, idx]
            ax.plot(rx, yo, "-", color=color, lw=1.1, label=f"{f:g} Hz obs" if fi == 0 or nfreq <= 4 else None)
            ax.plot(rx, ys, "--", color=color, lw=1.1, label=f"{f:g} Hz syn" if fi == 0 or nfreq <= 4 else None)
        ax.set_title(title)
        ax.set_ylabel("Phase (deg)" if is_phase else "Amplitude")
        if is_phase:
            ax.set_ylim(*_PHASE_YLIM)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Local rx index")
    axes[1, 1].set_xlabel("Local rx index")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), fontsize=8, frameon=False)
    fig.suptitle(f"Observed vs synthetic channel gains (Tx {tx_id}; solid=obs, dashed=syn)", y=0.98)
    return _save(fig, path)


def save_1d_section_figure(
    sec_x,
    sec_z,
    sec_rho,
    path: Path,
    *,
    sec_std=None,
    true_x=None,
    true_z=None,
    true_rho=None,
) -> Path:
    panels = [("1D mean", sec_x, sec_z, sec_rho, "viridis", "Ohm-m")]
    if sec_std is not None:
        std = np.asarray(sec_std, dtype=float)
        if np.any(np.isfinite(std)):
            panels.append(("1D std", sec_x, sec_z, std, "hot", "Std (Ohm-m)"))
    if true_rho is not None and true_x is not None and true_z is not None:
        panels.append(("True model", true_x, true_z, true_rho, "viridis", "Ohm-m"))
    rho_grids = [p[3] for p in panels if p[4] == "viridis"]
    vmin, vmax = _rho_limits(*rho_grids)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, x, z, grid, cmap, clabel) in zip(axes, panels):
        if cmap == "hot":
            lo = float(np.nanmin(grid)) if np.isfinite(grid).any() else 0.0
            hi = float(np.nanpercentile(np.asarray(grid)[np.isfinite(grid)], 98)) if np.isfinite(grid).any() else 1.0
            if hi <= lo:
                hi = lo + 1.0
            _heatmap(ax, x, z, grid, vmin=lo, vmax=hi, cmap=cmap, title=title, cbar_label=clabel)
        else:
            _heatmap(ax, x, z, grid, vmin=vmin, vmax=vmax, cmap=cmap, title=title, cbar_label=clabel)
    fig.suptitle("1D inversion pseudo-2D section", y=1.02)
    return _save(fig, path)


def save_1d_rho_vs_depth_figure(
    tx_ids: Sequence[int],
    tx_x: np.ndarray,
    rho_layers: np.ndarray,
    thickness_layers: np.ndarray,
    path: Path,
    *,
    z_start: Optional[float] = None,
    z_end: Optional[float] = None,
    tx_z: Optional[np.ndarray] = None,
    max_tx: int = 3,
) -> Path:
    tx_ids = np.asarray(tx_ids, dtype=int).reshape(-1)
    tx_x = np.asarray(tx_x, dtype=float).reshape(-1)
    rho_layers = np.asarray(rho_layers, dtype=float)
    thickness_layers = np.asarray(thickness_layers, dtype=float)
    if tx_ids.size == 0:
        raise ValueError("No transmitters in 1D results")
    picks = np.unique(np.round(np.linspace(0, tx_ids.size - 1, num=min(max_tx, tx_ids.size))).astype(int))
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for i in picks:
        rho = rho_layers[i]
        thk = thickness_layers[i]
        rho = rho[np.isfinite(rho)]
        thk = thk[np.isfinite(thk)]
        if rho.size == 0:
            continue
        n_layers = int(rho.size)
        thk = thk[: max(n_layers - 1, 0)]
        tz = float(tx_z[i]) if tx_z is not None and i < np.size(tx_z) else 0.0
        if z_start is not None:
            z0 = tz + float(z_start)
        else:
            z0 = tz
        depths = [z0]
        z_cur = z0
        for t in thk:
            z_cur = z_cur + float(t)
            depths.append(z_cur)
        if z_end is not None:
            depths.append(tz + float(z_end))
        else:
            depths.append(z_cur + (float(thk[-1]) if thk.size else 10.0))
        z_stairs = []
        rho_stairs = []
        for li in range(n_layers):
            za = depths[li]
            zb = depths[min(li + 1, len(depths) - 1)]
            z_stairs.extend([za, zb])
            rho_stairs.extend([float(rho[li]), float(rho[li])])
        label = f"Tx {int(tx_ids[i])} (x={tx_x[i]:g} m)" if i < tx_x.size else f"Tx {int(tx_ids[i])}"
        ax.plot(rho_stairs, z_stairs, lw=1.3, label=label)
    ax.set_xlabel("Resistivity (Ohm-m)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("1D inverted resistivity vs depth")
    return _save(fig, path)


def save_1d_obs_pred_figure(
    freqs,
    obs_hx,
    obs_hz,
    pred_hx,
    pred_hz,
    path: Path,
    *,
    tx_id: int,
    rx=None,
    c_per_freq=None,
) -> Path:
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    obs_hx = np.asarray(obs_hx, dtype=complex)
    obs_hz = np.asarray(obs_hz, dtype=complex)
    pred_hx = np.asarray(pred_hx, dtype=complex)
    pred_hz = np.asarray(pred_hz, dtype=complex)
    if c_per_freq is not None:
        c = np.asarray(c_per_freq, dtype=complex).reshape(-1, 1)
        pred_hx = c * pred_hx
        pred_hz = c * pred_hz
    nrx = int(obs_hx.shape[1]) if obs_hx.ndim == 2 else int(obs_hx.size)
    x = np.asarray(rx, dtype=float).reshape(-1) if rx is not None else np.arange(nrx, dtype=float)
    if x.size != nrx:
        x = np.arange(nrx, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2), sharex=True)
    cmap = plt.cm.tab10
    nfreq = int(min(freqs.size, obs_hx.shape[0], pred_hx.shape[0]))
    panels = (
        (axes[0, 0], obs_hx, pred_hx, False, "Hx amplitude"),
        (axes[0, 1], obs_hz, pred_hz, False, "Hz amplitude"),
        (axes[1, 0], obs_hx, pred_hx, True, "Hx phase (deg)"),
        (axes[1, 1], obs_hz, pred_hz, True, "Hz phase (deg)"),
    )
    for ax, obs, pred, is_phase, title in panels:
        for fi in range(nfreq):
            color = cmap(fi % 10)
            yo = np.angle(obs[fi], deg=True) if is_phase else np.abs(obs[fi])
            yp = np.angle(pred[fi], deg=True) if is_phase else np.abs(pred[fi])
            ax.plot(x, yo, "-", color=color, lw=1.1, label=f"{freqs[fi]:g} Hz obs" if fi == 0 or nfreq <= 4 else None)
            ax.plot(x, yp, "--", color=color, lw=1.1, label=f"{freqs[fi]:g} Hz pred" if fi == 0 or nfreq <= 4 else None)
        ax.set_title(title)
        ax.set_ylabel("Phase (deg)" if is_phase else "Amplitude")
        if is_phase:
            ax.set_ylim(*_PHASE_YLIM)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Receiver x (m)" if rx is not None else "Local rx index")
    axes[1, 1].set_xlabel("Receiver x (m)" if rx is not None else "Local rx index")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), fontsize=8, frameon=False)
    fig.suptitle(f"1D observed vs predicted gains (Tx {tx_id}; solid=obs, dashed=pred)", y=0.98)
    return _save(fig, path)


def save_1d_chi2_figure(
    tx_ids,
    chi2,
    path: Path,
    *,
    misfit=None,
    tx_x=None,
) -> Path:
    tx_ids = np.asarray(tx_ids, dtype=int).reshape(-1)
    chi2 = np.asarray(chi2, dtype=float).reshape(-1)
    x = np.asarray(tx_x, dtype=float).reshape(-1) if tx_x is not None else tx_ids.astype(float)
    if x.size != tx_ids.size:
        x = tx_ids.astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(x, chi2, "o-", lw=1.1, label=r"$\chi^2$")
    if misfit is not None:
        m = np.asarray(misfit, dtype=float).reshape(-1)
        if m.size == chi2.size:
            ax.plot(x, m, "s--", lw=1.0, label="Misfit")
    ax.set_xlabel("Tx x (m)" if tx_x is not None else "Tx index")
    ax.set_ylabel("Value")
    ax.set_title("1D inversion diagnostics per transmitter")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def conductivity_grid_to_resistivity(grid) -> np.ndarray:
    return conductivity_to_resistivity(grid)


__all__ = [
    "conductivity_grid_to_resistivity",
    "save_1d_chi2_figure",
    "save_1d_obs_pred_figure",
    "save_1d_rho_vs_depth_figure",
    "save_1d_section_figure",
    "save_2d_model_compare_figure",
    "save_2d_slices_figure",
    "save_amp_phase_vs_rx_figure",
    "save_calibration_c_figure",
    "save_obs_vs_syn_figure",
    "save_resistivity_survey_figure",
    "save_wavelet_figure",
    "survey_positions_from_meta",
]
