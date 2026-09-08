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
    dim=2,
    show_plot=True,
):
    """Create a continuous wave wavelet and write it to RSS."""
    params = compute_wavelet_parameters(flist=flist, dt=dt, n_periods=n_periods, rec_time=rec_time)

    wavmod = wavelet()
    wavmod.Ramp_sqw(alpha, params["nt"], params["dt"], params["flist"])
    wav = wavmod.wav
    if show_plot:
        wavmod.plot()

    dim_int = int(dim)
    if dim_int not in (2, 3):
        raise ValueError(f"Unsupported wavelet dim={dim_int}. Expected 2 or 3.")
    wav_out = rsfile(wav, dim_int)
    wav_out.geomD[0] = params["dt"]
    wav_out.write(wavfile)

    params["wavfile"] = wavfile
    params["lowpass_hz"] = params["freq"] * lowpass_factor
    params["alpha"] = alpha
    # `Ramp_sqw` ramps the CW source in over `alpha / flist[0]` seconds, i.e.
    # `alpha` periods of the LOWEST tone. `steady_state_phasor` deliberately
    # analyses only the LAST `n_periods` periods precisely so that this startup
    # transient is excluded - but that protection is defeated if the requested
    # window is the whole record, which is exactly what happens when
    # `n_periods_extract` equals the wavelet's own `n_periods`.
    #
    # Measured cost of getting this wrong on the workshop's own calibration
    # (order 6, homogeneous Earth, 1-6 kHz): |C|/dx^2 drifts 0.230 % across the
    # band with the full-record window, and 0.023 % once the ramp is excluded -
    # a factor of TEN, and it was masquerading as a residual physics error.
    #
    # `n_periods_extract_safe` is the largest INTEGER number of f_min periods
    # that clears the ramp. Integer matters too: a non-integer window (e.g. 4.5)
    # holds a non-integer number of periods of some tones and leaks between
    # them - measured 0.538 % drift at 4.5 against 0.023 % at 4.
    params["ramp_seconds"] = float(alpha) / float(params["flist"][0])
    params["ramp_periods_of_f_min"] = float(alpha)
    params["n_periods_extract_safe"] = float(
        max(1, int(np.floor(params["n_periods"] - np.ceil(float(alpha)))))
    )
    params["dim"] = dim_int
    params["waveform"] = np.asarray(wav).reshape(-1)
    params["time_axis_s"] = np.arange(params["nt"], dtype=float) * params["dt"]
    return params


__all__ = ["compute_wavelet_parameters", "create_wavelet_rss"]
