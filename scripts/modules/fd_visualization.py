"""FD output loading and amplitude/phase extraction helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from third_party.rockseis.io.rsfile import rsfile


def load_rss_traces(path):
    """
    Load an RSS shot record and return traces + geometry metadata.
    """
    path = Path(path)
    f = rsfile()
    f.read(str(path))

    data = np.asarray(f.data)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim > 2:
        data = data.reshape((data.shape[0], -1), order="F")

    nt = int(data.shape[0])
    ntrace = int(data.shape[1])
    dt = float(f.geomD[0])

    meta = {
        "path": str(path),
        "data": np.asarray(data, dtype=np.float64),
        "nt": nt,
        "ntrace": ntrace,
        "dt": dt,
        "src_x": np.asarray(getattr(f, "srcX", np.zeros(ntrace)), dtype=float)[:ntrace],
        "src_z": np.asarray(getattr(f, "srcZ", np.zeros(ntrace)), dtype=float)[:ntrace],
        "rx_x": np.asarray(getattr(f, "GroupX", np.zeros(ntrace)), dtype=float)[:ntrace],
        "rx_z": np.asarray(getattr(f, "GroupZ", np.zeros(ntrace)), dtype=float)[:ntrace],
    }
    return meta


def estimate_two_point(trace, dt, freq, t0):
    """
    Two-point amplitude/phase estimate aligned with Amp_and_fase.py.
    """
    t1 = t0 + 0.25 / freq
    idx0 = int(round(t0 / dt))
    idx1 = int(round(t1 / dt))
    if idx0 < 0 or idx1 >= len(trace):
        return np.nan, np.nan
    y0 = trace[idx0]
    y1 = trace[idx1]
    amp = np.hypot(y0, y1)
    theta = np.arctan2(-y1, y0)
    phi = theta - 2 * np.pi * freq * (idx0 * dt)
    phi = np.arctan2(np.sin(phi), np.cos(phi))
    return amp, phi


def estimate_two_point_window(trace, dt, freq, start_t, end_t, n_pairs):
    t0s = np.linspace(start_t, end_t - 0.25 / freq, num=int(n_pairs))
    amps = []
    phis = []
    for t0 in t0s:
        amp, phi = estimate_two_point(trace, dt, freq, t0)
        amps.append(amp)
        phis.append(phi)

    amps = np.asarray(amps, dtype=float)
    phis = np.asarray(phis, dtype=float)
    mean_amp = np.nanmean(amps)
    std_amp = np.nanstd(amps)
    mean_phi = np.angle(np.nanmean(np.exp(1j * phis)))
    phi_diffs = np.angle(np.exp(1j * (phis - mean_phi)))
    std_phi = np.nanstd(phi_diffs)
    return {
        "t0s": t0s,
        "amps": amps,
        "phis": phis,
        "mean_amp": mean_amp,
        "std_amp": std_amp,
        "mean_phi": mean_phi,
        "std_phi": std_phi,
    }


def build_trace_index(src_x, src_z, rx_x, rx_z, decimals=6):
    """
    Build tx/rx index arrays for each trace based on geometry coordinates.
    """
    src = np.column_stack(
        (
            np.round(np.asarray(src_x, dtype=float), decimals),
            np.round(np.asarray(src_z, dtype=float), decimals),
        )
    )
    rx = np.column_stack(
        (
            np.round(np.asarray(rx_x, dtype=float), decimals),
            np.round(np.asarray(rx_z, dtype=float), decimals),
        )
    )

    tx_unique, tx_idx = np.unique(src, axis=0, return_inverse=True)
    rx_unique, rx_idx = np.unique(rx, axis=0, return_inverse=True)

    # Local receiver index per tx: 0..(nrx-1) within each transmitter gather.
    rx_local_idx = np.full(rx_idx.shape, -1, dtype=int)
    nrx_per_tx = {}
    for tx_id in np.unique(tx_idx):
        tr = np.where(tx_idx == tx_id)[0]
        rx_ids_tx = rx_idx[tr]
        rx_unique_tx = np.unique(rx_ids_tx)
        map_local = {int(gid): i for i, gid in enumerate(rx_unique_tx)}
        rx_local_idx[tr] = np.asarray([map_local[int(g)] for g in rx_ids_tx], dtype=int)
        nrx_per_tx[int(tx_id)] = int(len(rx_unique_tx))

    return {
        "tx_unique": tx_unique,
        "rx_unique": rx_unique,
        "tx_idx_per_trace": tx_idx,
        "rx_idx_per_trace": rx_idx,
        "rx_local_idx_per_trace": rx_local_idx,
        "nrx_per_tx": nrx_per_tx,
    }


def compute_amp_phase_for_component(
    traces_data,
    dt,
    freqs,
    start_t,
    end_t,
    n_pairs=3,
):
    """
    Compute two-point window amplitude/phase for each (freq, trace).
    """
    traces = np.asarray(traces_data, dtype=float)
    if traces.ndim != 2:
        raise ValueError("traces_data must be 2D [nt, ntrace].")
    nt, ntrace = traces.shape
    freqs = np.asarray(freqs, dtype=float)

    amp_mean = np.full((len(freqs), ntrace), np.nan, dtype=float)
    amp_std = np.full((len(freqs), ntrace), np.nan, dtype=float)
    phi_mean = np.full((len(freqs), ntrace), np.nan, dtype=float)
    phi_std = np.full((len(freqs), ntrace), np.nan, dtype=float)

    for ifreq, freq in enumerate(freqs):
        if freq <= 0:
            continue
        start = float(start_t if start_t is not None else (1.0 / freq))
        stop = float(end_t if end_t is not None else (2.0 / freq))
        if stop <= start:
            stop = start + 0.25 / freq + dt
        for itr in range(ntrace):
            res = estimate_two_point_window(traces[:, itr], dt, freq, start, stop, n_pairs)
            amp_mean[ifreq, itr] = res["mean_amp"]
            amp_std[ifreq, itr] = res["std_amp"]
            phi_mean[ifreq, itr] = res["mean_phi"]
            phi_std[ifreq, itr] = res["std_phi"]

    return {
        "nt": nt,
        "ntrace": ntrace,
        "dt": float(dt),
        "freqs": freqs,
        "amp_mean": amp_mean,
        "amp_std": amp_std,
        "phi_mean_rad": phi_mean,
        "phi_std_rad": phi_std,
    }


def compute_amp_phase_for_fd_outputs(
    hx_path,
    hz_path,
    freqs,
    start_t=None,
    end_t=None,
    n_pairs=3,
):
    """
    End-to-end load + two-point extraction for Hx/Hz RSS outputs.
    """
    hx = load_rss_traces(hx_path)
    hz = load_rss_traces(hz_path)
    if hx["ntrace"] != hz["ntrace"]:
        raise ValueError(f"Hx/Hz trace count mismatch: {hx['ntrace']} vs {hz['ntrace']}")
    if abs(hx["dt"] - hz["dt"]) > 1e-12:
        raise ValueError(f"Hx/Hz dt mismatch: {hx['dt']} vs {hz['dt']}")

    idx = build_trace_index(hx["src_x"], hx["src_z"], hx["rx_x"], hx["rx_z"])
    hx_out = compute_amp_phase_for_component(
        hx["data"],
        dt=hx["dt"],
        freqs=freqs,
        start_t=start_t,
        end_t=end_t,
        n_pairs=n_pairs,
    )
    hz_out = compute_amp_phase_for_component(
        hz["data"],
        dt=hz["dt"],
        freqs=freqs,
        start_t=start_t,
        end_t=end_t,
        n_pairs=n_pairs,
    )
    return {
        "freqs": hx_out["freqs"],
        "geometry": {
            **idx,
            "src_x": hx["src_x"],
            "src_z": hx["src_z"],
            "rx_x": hx["rx_x"],
            "rx_z": hx["rx_z"],
            "ntrace": hx["ntrace"],
        },
        "Hx": hx_out,
        "Hz": hz_out,
    }


def save_amp_phase_npz(output_path, result):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geo = result["geometry"]
    hx = result["Hx"]
    hz = result["Hz"]
    np.savez(
        output_path,
        freqs=hx["freqs"],
        tx_idx_per_trace=geo["tx_idx_per_trace"],
        rx_idx_per_trace=geo["rx_idx_per_trace"],
        rx_local_idx_per_trace=geo["rx_local_idx_per_trace"],
        tx_unique=geo["tx_unique"],
        rx_unique=geo["rx_unique"],
        src_x=geo["src_x"],
        src_z=geo["src_z"],
        rx_x=geo["rx_x"],
        rx_z=geo["rx_z"],
        Hx_amp_mean=hx["amp_mean"],
        Hx_amp_std=hx["amp_std"],
        Hx_phi_mean_rad=hx["phi_mean_rad"],
        Hx_phi_std_rad=hx["phi_std_rad"],
        Hz_amp_mean=hz["amp_mean"],
        Hz_amp_std=hz["amp_std"],
        Hz_phi_mean_rad=hz["phi_mean_rad"],
        Hz_phi_std_rad=hz["phi_std_rad"],
    )


__all__ = [
    "build_trace_index",
    "compute_amp_phase_for_component",
    "compute_amp_phase_for_fd_outputs",
    "estimate_two_point",
    "estimate_two_point_window",
    "load_rss_traces",
    "save_amp_phase_npz",
]
