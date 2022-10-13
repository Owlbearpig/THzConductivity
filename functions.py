import matplotlib.pyplot as plt
import numpy as np
from imports import *
from numpy.fft import fft, fftfreq



def do_fft(t, y):
    dt = float(np.mean(np.diff(t)))
    freqs, data_fd = fftfreq(n=len(t), d=dt), fft(y)

    pos_slice = freqs >= 0

    return freqs[pos_slice], data_fd[pos_slice]


def phase_correction(data_fd, fit_range=None):
    if fit_range is None:
        fit_range = [0.25, 0.5]

    phase = np.angle(data_fd[:, 1])
    phase_unwrapped = np.unwrap(phase)

    fit_slice = (data_fd[:, 0] >= fit_range[0]) * (data_fd[:, 0] <= fit_range[1])
    p = np.polyfit(data_fd[fit_slice, 0], phase_unwrapped[fit_slice], 1)

    phase_corrected = phase_unwrapped - p[1].real

    plt.figure()
    plt.plot(data_fd[:, 0], phase_unwrapped, label="Unwrapped phase")
    plt.plot(data_fd[:, 0], phase_corrected, label="Shifted phase")
    plt.plot(data_fd[:, 0], data_fd[:, 0]*p[0].real, label="Lin. fit (slope*freq)")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (rad)")
    plt.legend()

    return phase_corrected


def windowing(data_td):
    peak_pos = np.argmax(data_td[:, 1])


