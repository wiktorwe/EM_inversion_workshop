import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt

from ..io.rsfile import rsfile


class wavelet(object):
    def __init__(self):
        self.wav = np.zeros([1])
        self.t = np.zeros([1])
        self.dt = 1.0
        self.wavcreated = 0

    def Ramp_sqw(self, alpha, Nt, dt, flist=[6e3, 12e3, 24e3, 48e3, 96e3]):
        t = np.linspace(0, (Nt - 1) * dt, Nt)
        sqw = np.zeros([Nt])
        for f in flist:
            sqw += np.sin(2 * np.pi * f * t)
        self.wav = np.zeros([Nt, 1])
        for i in range(0, Nt):
            if t[i] < 0:
                ramp = 0
            elif t[i] <= alpha / flist[0]:
                ramp = 0.5 * (1 - np.cos(2 * np.pi * flist[0] * t[i] / (2 * alpha)))
            else:
                ramp = 1
            self.wav[i, 0] = ramp * sqw[i]
        self.dt = dt
        self.wavcreated = 1
        self.t = t
        return self.wav, t

    def lowpass(self, fhi):
        if self.wavcreated:
            fs = 1 / self.dt
            sos = butter(10, fhi, "lowpass", fs=fs, output="sos")
            wav_filtered = sosfilt(sos, self.wav.squeeze())
            self.wav[:, 0] = wav_filtered

    def plot(self):
        if self.wavcreated:
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(self.t, self.wav)
            axs[0].grid()
            axs[0].set_xlabel("Time (s)")
            axs[0].set_ylabel("A")
            amp = np.real(np.abs(np.fft.fft(self.wav, axis=0)))
            freq = np.fft.fftfreq(self.t.size) / self.dt
            nwr = int(self.t.size / 2)
            axs[1].plot(freq[0:nwr], amp[0:nwr])
            axs[1].axis([freq[0], freq[nwr - 1], 0, 1.2 * np.max(amp)])
            axs[1].grid()
            axs[1].set_xlabel("Frequency (Hz)")
            axs[1].set_ylabel("|A|")
            plt.tight_layout()
            plt.show()

    def write(self, filename, dim):
        if self.wavcreated:
            wavfile = rsfile(self.wav, dim)
            wavfile.geomD[0] = self.dt
            wavfile.write(filename)
