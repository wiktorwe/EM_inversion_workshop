"""FD output loading and amplitude/phase extraction helpers.

Extraction used to time-differentiate Hx/Hz (`np.gradient`) and pick the
nearest FFT bin with an ad-hoc `|X|/(nt/4)` normalization
(`estimate_fft`/`compute_amp_phase_for_component`) - both deleted. Replaced
with the validated `steady_state_phasor` channel-gain-ratio pattern from
rockem-suite's own validation examples (`doc/examples/*/shared/phasor.py`,
via `scripts.modules.rockem_bridge`): apply the SAME Hanning-windowed
phasor extractor to the recorded trace AND the injected wavelet, take the
ratio. Any convention-dependent scale/window normalization cancels out of
that ratio - no time-derivative, no phase correction constants needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from third_party.rockseis.io.rsfile import rsfile

from scripts.modules.rockem_bridge import steady_state_phasor


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


def steady_state_gains(traces_data, dt, wavelet, wavelet_dt, freqs, f_min_hz, n_periods_extract=3.0):
    """Complex channel gain (`trace_phasor / wavelet_phasor`) per (freq, trace).

    Every frequency in `freqs` is extracted over the SAME absolute time
    window - `n_periods_extract` periods of `f_min_hz` - by scaling each
    tone's own `n_periods` argument to `steady_state_phasor` as
    `n_periods_extract * (freq / f_min_hz)`. This keeps a single shared
    Hann window across the whole multi-tone wavelet (holding an integer
    number of periods of every commensurate tone, e.g. 2/4/6 kHz sharing a
    window built from f_min=2 kHz), rather than a different, incompatible
    window per tone - the standard way to extract several simultaneous CW
    tones from one steady-state recording without them leaking into each
    other's estimate. The wavelet is windowed identically (same physical
    time span, own `wavelet_dt`), so the ratio cancels any convention-
    dependent scale - no time-derivative or phase-correction constants.
    """
    traces = np.asarray(traces_data, dtype=float)
    if traces.ndim != 2:
        raise ValueError("traces_data must be 2D [nt, ntrace].")
    nt, ntrace = traces.shape
    freqs = np.asarray(freqs, dtype=float)
    wavelet = np.asarray(wavelet, dtype=float).reshape(-1)

    gain = np.full((len(freqs), ntrace), np.nan, dtype=complex)
    for ifreq, freq in enumerate(freqs):
        if freq <= 0:
            continue
        n_periods_freq = n_periods_extract * (freq / f_min_hz)
        wav_phasor = steady_state_phasor(wavelet, wavelet_dt, freq, n_periods_freq)
        if wav_phasor == 0:
            continue
        for itr in range(ntrace):
            tr_phasor = steady_state_phasor(traces[:, itr], dt, freq, n_periods_freq)
            gain[ifreq, itr] = tr_phasor / wav_phasor

    return {
        "nt": nt,
        "ntrace": ntrace,
        "dt": float(dt),
        "freqs": freqs,
        "gain": gain,
        "amp_mean": np.abs(gain),
        "amp_std": np.zeros_like(np.abs(gain)),
        "phi_mean_rad": np.angle(gain),
        "phi_std_rad": np.zeros_like(np.angle(gain)),
    }


def compute_gains_for_fd_outputs(hx_path, hz_path, wavelet_path, freqs, f_min_hz, n_periods_extract=3.0):
    """End-to-end load + steady-state channel-gain extraction for Hx/Hz RSS
    outputs, referenced to the injected wavelet at `wavelet_path`."""
    hx = load_rss_traces(hx_path)
    hz = load_rss_traces(hz_path)
    wav = load_rss_traces(wavelet_path)
    if hx["ntrace"] != hz["ntrace"]:
        raise ValueError(f"Hx/Hz trace count mismatch: {hx['ntrace']} vs {hz['ntrace']}")
    if abs(hx["dt"] - hz["dt"]) > 1e-12:
        raise ValueError(f"Hx/Hz dt mismatch: {hx['dt']} vs {hz['dt']}")

    idx = build_trace_index(hx["src_x"], hx["src_z"], hx["rx_x"], hx["rx_z"])
    wavelet_trace = np.asarray(wav["data"], dtype=float)[:, 0]

    hx_out = steady_state_gains(
        hx["data"], dt=hx["dt"], wavelet=wavelet_trace, wavelet_dt=wav["dt"],
        freqs=freqs, f_min_hz=f_min_hz, n_periods_extract=n_periods_extract,
    )
    hz_out = steady_state_gains(
        hz["data"], dt=hz["dt"], wavelet=wavelet_trace, wavelet_dt=wav["dt"],
        freqs=freqs, f_min_hz=f_min_hz, n_periods_extract=n_periods_extract,
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
        "extraction": {
            "method": "steady_state_phasor_channel_gain",
            "f_min_hz": float(f_min_hz),
            "n_periods_extract": float(n_periods_extract),
            "wavelet_path": str(wavelet_path),
        },
    }


_PHASE_COMPARE_METRICS = frozenset(
    ("phase_vs_rx_deg", "phase_vs_tx_deg", "phase_vs_freq_deg")
)


def apply_compare_plot_yaxes(fig, metric, y_vals):
    """Set y-axis limits for real-vs-synthetic data-compare plots.

    Phase metrics stay on [-180, 180] deg. Amplitude metrics scale to the
    plotted data (anchored at 0) instead of forcing an upper limit of 1.0.
    """
    if metric in _PHASE_COMPARE_METRICS:
        fig.update_yaxes(range=[-180.0, 180.0])
        return

    y = np.asarray(y_vals, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        fig.update_yaxes(autorange=True, rangemode="tozero")
        return

    ymax = float(np.nanmax(y))
    ymin = float(np.nanmin(y))
    if ymax <= 0.0:
        fig.update_yaxes(range=[0.0, 1.0])
        return

    span = ymax - ymin
    pad = max(0.05 * span, 0.05 * ymax) if span > 0.0 else 0.05 * ymax
    fig.update_yaxes(range=[0.0, ymax + pad])


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
        Hx_gain=hx["gain"],
        Hz_amp_mean=hz["amp_mean"],
        Hz_amp_std=hz["amp_std"],
        Hz_phi_mean_rad=hz["phi_mean_rad"],
        Hz_phi_std_rad=hz["phi_std_rad"],
        Hz_gain=hz["gain"],
    )


__all__ = [
    "apply_compare_plot_yaxes",
    "build_trace_index",
    "steady_state_gains",
    "compute_gains_for_fd_outputs",
    "load_rss_traces",
    "save_amp_phase_npz",
]
