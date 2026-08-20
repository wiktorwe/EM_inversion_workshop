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
_MAX_RX_PANELS = 8


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _fig_axes(nrows, ncols, figsize, *, sharex=False, sharey=False):
    return plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        layout="constrained",
        sharex=sharex,
        sharey=sharey,
    )


def _legend_below(fig, handles, labels, ncol: int = 4):
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(ncol, max(len(labels), 1)),
        fontsize=8,
        frameon=False,
        borderaxespad=0.2,
    )


def _apply_phase_ylim(ax):
    """Zoom clustered phase traces instead of always using ±180 deg."""
    ys = []
    for line in ax.get_lines():
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y)]
        if y.size:
            ys.append(y)
    if not ys:
        ax.set_ylim(*_PHASE_YLIM)
        return
    stacked = np.concatenate(ys)
    lo, hi = float(np.min(stacked)), float(np.max(stacked))
    span = max(hi - lo, 8.0)
    if span >= 160.0:
        ax.set_ylim(*_PHASE_YLIM)
        return
    pad = 0.35 * span
    ax.set_ylim(lo - pad, hi + pad)


def _vs_tx_figsize(nrx: int) -> tuple[float, float]:
    # Two rows per receiver (amp then phase) keeps 2-col panels readable at text width.
    return (9.4, 4.3 * int(nrx) + 1.15)


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


def _heatmap(ax, x, z, rho, *, vmin, vmax, cmap="jet", title="", cbar_label="Ohm-m", colorbar=True):
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    rho = np.asarray(rho, dtype=float)
    im = ax.pcolormesh(
        x, z, rho, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True
    )
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    if title:
        ax.set_title(title)
    ax.set_aspect("auto")
    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label(cbar_label)
    return im


def _depth_down(ax):
    """Put larger depth at the bottom. Safe to call once on a shared y-axis."""
    lo, hi = ax.get_ylim()
    if lo < hi:
        ax.invert_yaxis()


def _scatter_survey(ax, tx_x=None, tx_z=None, rx_x=None, rx_z=None, *, legend=False):
    handles = []
    if tx_x is not None and tx_z is not None and np.size(tx_x):
        h = ax.scatter(tx_x, tx_z, c="red", marker="x", s=36, linewidths=1.2, label="Sources", zorder=3)
        handles.append(h)
    if rx_x is not None and rx_z is not None and np.size(rx_x):
        rec_step = max(1, int(np.ceil(np.size(rx_x) / 5000)))
        h = ax.scatter(
            np.asarray(rx_x)[::rec_step],
            np.asarray(rx_z)[::rec_step],
            c="white",
            s=8,
            edgecolors="k",
            linewidths=0.3,
            label="Receivers",
            zorder=2,
        )
        handles.append(h)
    if legend and handles:
        ax.legend(handles=handles, loc="upper right", framealpha=0.85, fontsize=8)
    return handles


def _mid_index(values: np.ndarray) -> int:
    vals = np.unique(np.asarray(values, dtype=int))
    if vals.size == 0:
        return 0
    return int(vals[vals.size // 2])


def _pick_ids(ids: np.ndarray, max_n: int) -> np.ndarray:
    ids = np.unique(np.asarray(ids, dtype=int))
    if ids.size <= max_n:
        return ids
    picks = np.unique(np.round(np.linspace(0, ids.size - 1, num=max_n)).astype(int))
    return ids[picks]


def _freq_list(gains: Mapping) -> np.ndarray:
    return np.asarray(gains.get("freqs", gains.get("Hx", {}).get("freqs", [])), dtype=float)


def _rx_local(geo: Mapping) -> np.ndarray:
    return np.asarray(geo.get("rx_local_idx_per_trace", geo.get("rx_idx_per_trace")), dtype=int)


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
    fig, ax = _fig_axes(1, 1, (8.4, 4.6))
    _heatmap(ax, x, z, rho, vmin=vmin, vmax=vmax, cmap="jet", title=title)
    _scatter_survey(ax, tx_x, tx_z, rx_x, rx_z, legend=True)
    _depth_down(ax)
    return _save(fig, path)


def save_wavelet_figure(t, w, path: Path, *, flist_hz: Optional[Sequence[float]] = None) -> Path:
    """Time-domain wavelet and amplitude spectrum zoomed around the design frequencies.

    The stored ``wav2d.rss`` is often interpolated onto the modelling ``dt``
    (~1e-8 s), so an unzoomed FFT axis reaches tens of MHz and collapses the
    kHz design lines into a single stem at the origin.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    n = int(min(t.size, w.size))
    t, w = t[:n], w[:n]
    dt = float(t[1] - t[0]) if n > 1 else 1.0
    freqs = np.fft.rfftfreq(n, d=max(dt, 1e-30))
    amp = np.abs(np.fft.rfft(w))
    fig, axes = _fig_axes(1, 2, (9.2, 3.4))
    axes[0].plot(t, w, color="C0", lw=1.0)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Wavelet (time domain)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(freqs, amp, color="C0", lw=1.0)
    flist = np.asarray([] if flist_hz is None else flist_hz, dtype=float)
    flist = flist[np.isfinite(flist) & (flist > 0)]
    nyquist = float(freqs[-1]) if freqs.size else 0.0
    if flist.size:
        fmax = float(np.max(flist))
        xmax = min(nyquist, max(2.5 * fmax, fmax + 500.0))
        for f0 in flist:
            axes[1].axvline(float(f0), color="0.55", ls=":", lw=0.9, zorder=0)
        axes[1].set_xlim(0.0, xmax)
    elif nyquist > 0:
        # Keep the lowest 2% of Nyquist so kHz content is visible on MHz axes.
        axes[1].set_xlim(0.0, max(nyquist * 0.02, 1.0))
    axes[1].ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("|FFT|")
    axes[1].set_title("Wavelet amplitude spectrum")
    axes[1].grid(True, alpha=0.3)
    return _save(fig, path)


def _trace_subset(gains: Mapping, tx_id: int) -> tuple[np.ndarray, np.ndarray]:
    geo = gains["geometry"]
    tx = np.asarray(geo["tx_idx_per_trace"], dtype=int)
    rx_local = _rx_local(geo)
    idx = np.where(tx == int(tx_id))[0]
    if idx.size == 0:
        raise ValueError(f"No traces for transmitter {tx_id}")
    order = np.argsort(rx_local[idx])
    return idx[order], rx_local[idx][order]


def _tx_subset(gains: Mapping, rx_id: int) -> tuple[np.ndarray, np.ndarray]:
    geo = gains["geometry"]
    tx = np.asarray(geo["tx_idx_per_trace"], dtype=int)
    rx_local = _rx_local(geo)
    idx = np.where(rx_local == int(rx_id))[0]
    if idx.size == 0:
        raise ValueError(f"No traces for receiver {rx_id}")
    order = np.argsort(tx[idx])
    return idx[order], tx[idx][order]


def save_amp_phase_vs_rx_figure(gains: Mapping, path: Path, *, tx_id: Optional[int] = None) -> Path:
    """Hx/Hz amplitude and phase vs local rx, all frequencies overplotted."""
    geo = gains["geometry"]
    tx = np.asarray(geo["tx_idx_per_trace"], dtype=int)
    if tx_id is None:
        tx_id = _mid_index(tx)
    idx, rx = _trace_subset(gains, tx_id)
    freqs = _freq_list(gains)
    fig, axes = _fig_axes(2, 2, (9.2, 6.6), sharex=True)
    panels = (
        (axes[0, 0], "Hx", "amp_mean", "Hx amplitude", False),
        (axes[0, 1], "Hz", "amp_mean", "Hz amplitude", False),
        (axes[1, 0], "Hx", "phi_mean_rad", "Hx phase", True),
        (axes[1, 1], "Hz", "phi_mean_rad", "Hz phase", True),
    )
    for ax, comp, key, title, is_phase in panels:
        arr = np.asarray(gains[comp][key], dtype=float)
        for fi, f in enumerate(freqs):
            y = arr[fi, idx]
            if is_phase:
                y = np.rad2deg(y)
            ax.plot(rx, y, lw=1.0, marker="o", ms=3.0, label=f"{f:g} Hz")
        ax.set_title(title)
        if is_phase:
            _apply_phase_ylim(ax)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel("Phase (deg)")
    axes[1, 0].set_xlabel("Local rx index")
    axes[1, 1].set_xlabel("Local rx index")
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.04, wspace=0.06, hspace=0.08)
    _legend_below(fig, *axes[0, 0].get_legend_handles_labels())
    fig.suptitle(f"Channel-gain amp/phase vs rx (Tx {tx_id})", fontsize=11)
    return _save(fig, path)


def save_amp_phase_vs_tx_figure(gains: Mapping, path: Path, *, max_rx: int = _MAX_RX_PANELS) -> Path:
    """Hx/Hz amplitude and phase vs Tx, one row of panels per receiver."""
    geo = gains["geometry"]
    rx_ids = _pick_ids(_rx_local(geo), max_rx)
    if rx_ids.size == 0:
        raise ValueError("No receiver indices in modelled-data geometry")
    freqs = _freq_list(gains)
    nrx = int(rx_ids.size)
    fig, axes = _fig_axes(2 * nrx, 2, _vs_tx_figsize(nrx), sharex=True)
    axes = np.atleast_2d(axes)
    specs = (
        (0, "Hx", "amp_mean", False, "Hx amplitude"),
        (1, "Hz", "amp_mean", False, "Hz amplitude"),
        (0, "Hx", "phi_mean_rad", True, "Hx phase"),
        (1, "Hz", "phi_mean_rad", True, "Hz phase"),
    )
    for ri, rx_id in enumerate(rx_ids):
        idx, tx = _tx_subset(gains, int(rx_id))
        for ci, comp, key, is_phase, title in specs:
            ax = axes[2 * ri + int(is_phase), ci]
            arr = np.asarray(gains[comp][key], dtype=float)
            for fi, f in enumerate(freqs):
                y = np.rad2deg(arr[fi, idx]) if is_phase else arr[fi, idx]
                ax.plot(tx, y, lw=1.0, marker="o", ms=3.0, label=f"{f:g} Hz")
            if is_phase:
                _apply_phase_ylim(ax)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            if ri == 0:
                ax.set_title(title)
        axes[2 * ri, 0].set_ylabel(f"Rx {int(rx_id)}\nAmplitude")
        axes[2 * ri + 1, 0].set_ylabel("Phase (deg)")
    axes[-1, 0].set_xlabel("Tx index")
    axes[-1, 1].set_xlabel("Tx index")
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.04, wspace=0.08, hspace=0.10)
    _legend_below(fig, *axes[0, 0].get_legend_handles_labels())
    fig.suptitle("Channel-gain amp/phase vs Tx (amp and phase rows per receiver)", fontsize=11)
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
    fig, axes = _fig_axes(1, ncols, (3.15 * ncols + 1.0, 3.6))
    axes = np.atleast_1d(axes)
    mag = np.abs(c_arr)
    axes[0].plot(freqs, mag, "o-", lw=1.1)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("|C|")
    axes[0].set_title("|C(f)|")
    axes[0].grid(True, alpha=0.3)
    if mag.size and np.nanmax(mag) - np.nanmin(mag) < 0.05 * max(np.nanmean(mag), 1e-12):
        pad = 0.01 * max(float(np.nanmean(mag)), 1e-6)
        axes[0].set_ylim(float(np.nanmin(mag)) - pad, float(np.nanmax(mag)) + pad)
    phi = np.angle(c_arr, deg=True)
    axes[1].plot(freqs, phi, "o-", lw=1.1, color="C1")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (deg)")
    axes[1].set_title("arg C(f)")
    axes[1].grid(True, alpha=0.3)
    if phi.size:
        span = float(max(np.nanmax(phi) - np.nanmin(phi), 0.5))
        axes[1].set_ylim(float(np.nanmin(phi)) - 0.4 * span, float(np.nanmax(phi)) + 0.4 * span)
    if has_scatter:
        hx = None if scatter_hx_pct is None else np.asarray(scatter_hx_pct, dtype=float)[:n]
        hz = None if scatter_hz_pct is None else np.asarray(scatter_hz_pct, dtype=float)[:n]
        if hx is not None:
            axes[2].plot(freqs, hx, "o-", label="Hx")
        if hz is not None:
            axes[2].plot(freqs, hz, "s-", label="Hz")
        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].set_ylabel("Relative scatter (%)")
        axes[2].set_title("Calibration scatter")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
        stacked = [a for a in (hx, hz) if a is not None and np.any(np.isfinite(a) & (a > 0))]
        if stacked:
            lo = min(float(np.nanmin(a[a > 0])) for a in stacked)
            hi = max(float(np.nanmax(a)) for a in stacked)
            if hi / max(lo, 1e-12) > 50.0:
                axes[2].set_yscale("log")
    title = "FDTD–analytic calibration C(f)"
    if method:
        title += f" ({method})"
    fig.suptitle(title, fontsize=11)
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
    tx_x=None,
    tx_z=None,
    rx_x=None,
    rx_z=None,
) -> Path:
    vmin, vmax = _rho_limits(rho_true, rho_inv)
    fig, axes = _fig_axes(1, 2, (9.6, 4.2), sharey=True)
    im0 = _heatmap(axes[0], x_true, z_true, rho_true, vmin=vmin, vmax=vmax, cmap="viridis", title="True model", colorbar=False)
    _heatmap(axes[1], x_inv, z_inv, rho_inv, vmin=vmin, vmax=vmax, cmap="viridis", title=inv_label, colorbar=False)
    _scatter_survey(axes[0], tx_x, tx_z, rx_x, rx_z, legend=True)
    _scatter_survey(axes[1], tx_x, tx_z, rx_x, rx_z, legend=False)
    _depth_down(axes[0])
    fig.colorbar(im0, ax=axes, shrink=0.82, pad=0.02, label="Ohm-m")
    fig.suptitle("2D resistivity: true vs inverted", fontsize=11)
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

    fig, axes = _fig_axes(1, 2, (8.8, 3.8))
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
    freqs = _freq_list(obs)
    nfreq = int(freqs.size)
    fig, axes = _fig_axes(2, 2, (9.2, 6.6), sharex=True)
    specs = (
        (axes[0, 0], "Hx", "amp_mean", "Hx amplitude", False),
        (axes[0, 1], "Hz", "amp_mean", "Hz amplitude", False),
        (axes[1, 0], "Hx", "phi_mean_rad", "Hx phase", True),
        (axes[1, 1], "Hz", "phi_mean_rad", "Hz phase", True),
    )
    cmap = plt.cm.tab10
    for ax, comp, key, title, is_phase in specs:
        o = np.asarray(obs[comp][key], dtype=float)
        s = np.asarray(syn[comp][key], dtype=float)
        for fi, f in enumerate(freqs):
            color = cmap(fi % 10)
            yo = np.rad2deg(o[fi, idx]) if is_phase else o[fi, idx]
            ys = np.rad2deg(s[fi, idx]) if is_phase else s[fi, idx]
            ax.plot(rx, yo, "-", color=color, lw=1.1, label=f"{f:g} Hz obs" if nfreq <= 6 else None)
            ax.plot(rx, ys, "--", color=color, lw=1.1, label=f"{f:g} Hz syn" if nfreq <= 6 else None)
        ax.set_title(title)
        if is_phase:
            _apply_phase_ylim(ax)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel("Phase (deg)")
    axes[1, 0].set_xlabel("Local rx index")
    axes[1, 1].set_xlabel("Local rx index")
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.04, wspace=0.06, hspace=0.08)
    _legend_below(fig, *axes[0, 0].get_legend_handles_labels(), ncol=4)
    fig.suptitle(f"Observed vs synthetic vs rx (Tx {tx_id}; solid=obs, dashed=syn)", fontsize=11)
    return _save(fig, path)


def save_obs_vs_syn_vs_tx_figure(
    obs: Mapping,
    syn: Mapping,
    path: Path,
    *,
    max_rx: int = _MAX_RX_PANELS,
) -> Path:
    rx_ids = _pick_ids(_rx_local(obs["geometry"]), max_rx)
    if rx_ids.size == 0:
        raise ValueError("No receiver indices for observed-vs-synthetic vs Tx")
    freqs = _freq_list(obs)
    nrx = int(rx_ids.size)
    fig, axes = _fig_axes(2 * nrx, 2, _vs_tx_figsize(nrx), sharex=True)
    axes = np.atleast_2d(axes)
    specs = (
        (0, "Hx", "amp_mean", False, "Hx amplitude"),
        (1, "Hz", "amp_mean", False, "Hz amplitude"),
        (0, "Hx", "phi_mean_rad", True, "Hx phase"),
        (1, "Hz", "phi_mean_rad", True, "Hz phase"),
    )
    cmap = plt.cm.tab10
    for ri, rx_id in enumerate(rx_ids):
        idx, tx = _tx_subset(obs, int(rx_id))
        for ci, comp, key, is_phase, title in specs:
            ax = axes[2 * ri + int(is_phase), ci]
            o = np.asarray(obs[comp][key], dtype=float)
            s = np.asarray(syn[comp][key], dtype=float)
            for fi, f in enumerate(freqs):
                color = cmap(fi % 10)
                yo = np.rad2deg(o[fi, idx]) if is_phase else o[fi, idx]
                ys = np.rad2deg(s[fi, idx]) if is_phase else s[fi, idx]
                ax.plot(tx, yo, "-", color=color, lw=1.1, marker="o", ms=3.0, label=f"{f:g} Hz obs")
                ax.plot(tx, ys, "--", color=color, lw=1.1, marker="s", ms=2.5, label=f"{f:g} Hz syn")
            if is_phase:
                _apply_phase_ylim(ax)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            if ri == 0:
                ax.set_title(title)
        axes[2 * ri, 0].set_ylabel(f"Rx {int(rx_id)}\nAmplitude")
        axes[2 * ri + 1, 0].set_ylabel("Phase (deg)")
    axes[-1, 0].set_xlabel("Tx index")
    axes[-1, 1].set_xlabel("Tx index")
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.04, wspace=0.08, hspace=0.10)
    _legend_below(fig, *axes[0, 0].get_legend_handles_labels(), ncol=4)
    fig.suptitle("Observed vs synthetic vs Tx (amp and phase rows per receiver; solid=obs, dashed=syn)", fontsize=11)
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
    tx_x=None,
    tx_z=None,
    rx_x=None,
    rx_z=None,
) -> Path:
    panels = [("1D mean", sec_x, sec_z, sec_rho, "viridis", "Ohm-m")]
    if sec_std is not None:
        std = np.asarray(sec_std, dtype=float)
        finite = std[np.isfinite(std)]
        if finite.size and float(np.nanmax(np.abs(finite))) > 1e-12:
            panels.append(("1D std", sec_x, sec_z, std, "hot", "Std (Ohm-m)"))
    if true_rho is not None and true_x is not None and true_z is not None:
        panels.append(("True model", true_x, true_z, true_rho, "viridis", "Ohm-m"))
    rho_grids = [p[3] for p in panels if p[4] == "viridis"]
    vmin, vmax = _rho_limits(*rho_grids)
    n = len(panels)
    fig, axes = _fig_axes(1, n, (4.4 * n, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    viridis_ims = []
    std_im = None
    for ax, (title, x, z, grid, cmap, clabel) in zip(axes, panels):
        if cmap == "hot":
            finite = np.asarray(grid, dtype=float)
            finite = finite[np.isfinite(finite)]
            lo = float(np.nanmin(finite)) if finite.size else 0.0
            hi = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
            if hi <= lo:
                hi = lo + 1.0
            std_im = _heatmap(ax, x, z, grid, vmin=lo, vmax=hi, cmap=cmap, title=title, colorbar=False)
            ax.set_ylabel("")
        else:
            im = _heatmap(ax, x, z, grid, vmin=vmin, vmax=vmax, cmap=cmap, title=title, colorbar=False)
            viridis_ims.append(im)
            if ax is not axes[0]:
                ax.set_ylabel("")
    _depth_down(axes[0])
    legend_done = False
    for ax, (_, _, _, _, cmap, _) in zip(axes, panels):
        if cmap == "hot":
            continue
        _scatter_survey(ax, tx_x, tx_z, rx_x, rx_z, legend=not legend_done)
        legend_done = True
    if viridis_ims:
        fig.colorbar(viridis_ims[-1], ax=[ax for ax, p in zip(axes, panels) if p[4] == "viridis"], shrink=0.82, pad=0.02, label="Ohm-m")
    if std_im is not None:
        std_axes = [ax for ax, p in zip(axes, panels) if p[4] == "hot"]
        fig.colorbar(std_im, ax=std_axes, shrink=0.82, pad=0.02, label="Std (Ohm-m)")
    fig.suptitle("1D inversion pseudo-2D section", fontsize=11)
    return _save(fig, path)


def _absolute_depth_window(z_start, z_end, tz: float) -> bool:
    if z_start is None or z_end is None:
        return False
    zs, ze = float(z_start), float(z_end)
    if ze <= zs:
        return False
    return zs >= 0.0 and (zs <= tz <= ze or zs > 100.0)


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
    fig, ax = _fig_axes(1, 1, (5.8, 4.8))
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
        if _absolute_depth_window(z_start, z_end, tz):
            z0 = float(z_start)
            z_bottom = float(z_end)
        else:
            z0 = tz + (float(z_start) if z_start is not None else 0.0)
            z_bottom = tz + (float(z_end) if z_end is not None else z0 + float(np.sum(thk) if thk.size else 10.0))
        depths = [z0]
        z_cur = z0
        for tval in thk:
            z_cur = z_cur + float(tval)
            depths.append(z_cur)
        depths.append(z_bottom)
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
    ax.legend(fontsize=8, loc="best")
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
    fig, axes = _fig_axes(2, 2, (9.2, 6.6), sharex=True)
    cmap = plt.cm.tab10
    nfreq = int(min(freqs.size, obs_hx.shape[0], pred_hx.shape[0]))
    panels = (
        (axes[0, 0], obs_hx, pred_hx, False, "Hx amplitude"),
        (axes[0, 1], obs_hz, pred_hz, False, "Hz amplitude"),
        (axes[1, 0], obs_hx, pred_hx, True, "Hx phase"),
        (axes[1, 1], obs_hz, pred_hz, True, "Hz phase"),
    )
    for ax, obs, pred, is_phase, title in panels:
        for fi in range(nfreq):
            color = cmap(fi % 10)
            yo = np.angle(obs[fi], deg=True) if is_phase else np.abs(obs[fi])
            yp = np.angle(pred[fi], deg=True) if is_phase else np.abs(pred[fi])
            ax.plot(x, yo, "-", color=color, lw=1.1, marker="o", ms=3.0, label=f"{freqs[fi]:g} Hz obs")
            ax.plot(x, yp, "--", color=color, lw=1.1, marker="s", ms=2.5, label=f"{freqs[fi]:g} Hz pred")
        ax.set_title(title)
        if is_phase:
            _apply_phase_ylim(ax)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel("Phase (deg)")
    xlabel = "Receiver x (m)" if rx is not None and np.asarray(rx).size == nrx else "Local rx index"
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 1].set_xlabel(xlabel)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.04, wspace=0.06, hspace=0.08)
    _legend_below(fig, *axes[0, 0].get_legend_handles_labels(), ncol=4)
    fig.suptitle(f"1D observed vs predicted vs rx (Tx {tx_id}; solid=obs, dashed=pred)", fontsize=11)
    return _save(fig, path)


def save_1d_obs_pred_vs_tx_figure(
    data: Mapping,
    path: Path,
    *,
    freqs=None,
    c_per_freq=None,
    max_rx: int = _MAX_RX_PANELS,
) -> Path:
    """Amp/phase vs Tx for each local Rx from a 1D inversion NPZ."""
    tx_ids = np.asarray(data.get("tx_ids", []), dtype=int).reshape(-1)
    if tx_ids.size == 0:
        raise ValueError("No transmitters in 1D NPZ")
    sample = None
    for tid in tx_ids:
        if f"obs_hx_gain_tx{int(tid)}" in data:
            sample = np.asarray(data[f"obs_hx_gain_tx{int(tid)}"], dtype=complex)
            break
    if sample is None:
        raise ValueError("1D NPZ has no obs_hx_gain_tx* arrays")
    nfreq, nrx = int(sample.shape[0]), int(sample.shape[1]) if sample.ndim == 2 else 1
    if freqs is None:
        freqs = data.get(f"freqs_tx{int(tx_ids[0])}")
    freqs = np.asarray(freqs if freqs is not None else np.arange(nfreq), dtype=float).reshape(-1)
    nfreq = int(min(nfreq, freqs.size))
    rx_ids = _pick_ids(np.arange(nrx, dtype=int), max_rx)

    def _stack(key_fmt: str) -> np.ndarray:
        out = np.full((nfreq, tx_ids.size, nrx), np.nan, dtype=complex)
        for j, tid in enumerate(tx_ids):
            arr = data.get(key_fmt.format(int(tid)))
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=complex)
            if arr.ndim == 1:
                arr = arr.reshape(nfreq, 1)
            out[:, j, : min(nrx, arr.shape[1])] = arr[:nfreq, :nrx]
        if c_per_freq is not None and "pred" in key_fmt:
            c = np.asarray(c_per_freq, dtype=complex).reshape(-1, 1, 1)[:nfreq]
            out = c * out
        return out

    obs_hx = _stack("obs_hx_gain_tx{}")
    obs_hz = _stack("obs_hz_gain_tx{}")
    pred_hx = _stack("pred_hxh_mean_tx{}")
    pred_hz = _stack("pred_hxhz_mean_tx{}")
    x_tx = np.asarray(data.get("tx_x", tx_ids), dtype=float).reshape(-1)
    if x_tx.size != tx_ids.size:
        x_tx = tx_ids.astype(float)
    xlabel = "Tx x (m)" if "tx_x" in data else "Tx index"
    nrx_plot = int(rx_ids.size)
    fig, axes = _fig_axes(2 * nrx_plot, 2, _vs_tx_figsize(nrx_plot), sharex=True)
    axes = np.atleast_2d(axes)
    cmap = plt.cm.tab10
    panels = (
        (0, obs_hx, pred_hx, False, "Hx amplitude"),
        (1, obs_hz, pred_hz, False, "Hz amplitude"),
        (0, obs_hx, pred_hx, True, "Hx phase"),
        (1, obs_hz, pred_hz, True, "Hz phase"),
    )
    for ri, rx_id in enumerate(rx_ids):
        for ci, obs, pred, is_phase, title in panels:
            ax = axes[2 * ri + int(is_phase), ci]
            for fi in range(nfreq):
                color = cmap(fi % 10)
                yo = np.angle(obs[fi, :, int(rx_id)], deg=True) if is_phase else np.abs(obs[fi, :, int(rx_id)])
                yp = np.angle(pred[fi, :, int(rx_id)], deg=True) if is_phase else np.abs(pred[fi, :, int(rx_id)])
                ax.plot(x_tx, yo, "-", color=color, lw=1.1, marker="o", ms=3.0, label=f"{freqs[fi]:g} Hz obs")
                ax.plot(x_tx, yp, "--", color=color, lw=1.1, marker="s", ms=2.5, label=f"{freqs[fi]:g} Hz pred")
            if is_phase:
                _apply_phase_ylim(ax)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            if ri == 0:
                ax.set_title(title)
        axes[2 * ri, 0].set_ylabel(f"Rx {int(rx_id)}\nAmplitude")
        axes[2 * ri + 1, 0].set_ylabel("Phase (deg)")
    axes[-1, 0].set_xlabel(xlabel)
    axes[-1, 1].set_xlabel(xlabel)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.04, wspace=0.08, hspace=0.10)
    _legend_below(fig, *axes[0, 0].get_legend_handles_labels(), ncol=4)
    fig.suptitle("1D observed vs predicted vs Tx (amp and phase rows per receiver; solid=obs, dashed=pred)", fontsize=11)
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
    fig, ax = _fig_axes(1, 1, (7.4, 3.6))
    h1 = ax.plot(x, chi2, "o-", color="C0", lw=1.1, label="chi2")
    ax.set_xlabel("Tx x (m)" if tx_x is not None else "Tx index")
    ax.set_ylabel("chi2", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_title("1D inversion diagnostics per transmitter")
    ax.grid(True, alpha=0.3)
    handles = list(h1)
    if misfit is not None:
        m = np.asarray(misfit, dtype=float).reshape(-1)
        if m.size == chi2.size:
            ax2 = ax.twinx()
            h2 = ax2.plot(x, m, "s--", color="C1", lw=1.0, label="Misfit")
            ax2.set_ylabel("Misfit", color="C1")
            ax2.tick_params(axis="y", labelcolor="C1")
            handles.extend(h2)
    ax.legend(handles, [h.get_label() for h in handles], fontsize=8, loc="best")
    return _save(fig, path)


def conductivity_grid_to_resistivity(grid) -> np.ndarray:
    return conductivity_to_resistivity(grid)


__all__ = [
    "conductivity_grid_to_resistivity",
    "save_1d_chi2_figure",
    "save_1d_obs_pred_figure",
    "save_1d_obs_pred_vs_tx_figure",
    "save_1d_rho_vs_depth_figure",
    "save_1d_section_figure",
    "save_2d_model_compare_figure",
    "save_2d_slices_figure",
    "save_amp_phase_vs_rx_figure",
    "save_amp_phase_vs_tx_figure",
    "save_calibration_c_figure",
    "save_obs_vs_syn_figure",
    "save_obs_vs_syn_vs_tx_figure",
    "save_resistivity_survey_figure",
    "save_wavelet_figure",
    "survey_positions_from_meta",
]
