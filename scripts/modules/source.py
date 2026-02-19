"""Wavelet-source helpers for workshop workflows."""

import numpy as np

from third_party.rockseis.io.rsfile import rsfile
from third_party.rockseis.wavelet.wavelet import wavelet


def compute_wavelet_parameters(flist, dt, n_periods=None, rec_time=10e-9):
    """Compute wavelet sampling parameters from input settings."""
    if not flist:
        flist = [6e3, 12e3, 24e3, 48e3, 96e3]

    freq = flist[0]
    period = 1.0 / freq

    if n_periods is not None:
        n_periods_eff = max(1, int(n_periods))
    else:
        n_periods_eff = max(1, int(np.round(rec_time / period)))

    rec_time_actual = n_periods_eff * period
    nt = int(np.floor(rec_time_actual / dt)) + 1

    return {
        "flist": list(flist),
        "freq": freq,
        "period": period,
        "n_periods": n_periods_eff,
        "rec_time_actual": rec_time_actual,
        "dt": dt,
        "nt": nt,
        "samples_per_period": int(period / dt),
    }


def create_wavelet_rss(
    flist=None,
    dt=1.13021e-11,
    n_periods=None,
    rec_time=10e-9,
    lowpass_factor=1.5,
    alpha=0.5,
    wavfile="wav2d.rss",
    show_plot=True,
):
    """Create a continuous wave wavelet and write it to RSS."""
    params = compute_wavelet_parameters(flist=flist, dt=dt, n_periods=n_periods, rec_time=rec_time)

    wavmod = wavelet()
    wavmod.Ramp_sqw(alpha, params["nt"], params["dt"], params["flist"])
    wav = wavmod.wav
    if show_plot:
        wavmod.plot()

    wav_out = rsfile(wav, 2)
    wav_out.geomD[0] = params["dt"]
    wav_out.write(wavfile)

    params["wavfile"] = wavfile
    params["lowpass_hz"] = params["freq"] * lowpass_factor
    params["alpha"] = alpha
    params["waveform"] = np.asarray(wav).reshape(-1)
    params["time_axis_s"] = np.arange(params["nt"], dtype=float) * params["dt"]
    return params


__all__ = ["compute_wavelet_parameters", "create_wavelet_rss"]
