import matplotlib.pyplot as plt
import numpy as np
from imports import *
from numpy.fft import fft, fftfreq
from scipy import signal


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
    plt.plot(data_fd[:, 0], data_fd[:, 0] * p[0].real, label="Lin. fit (slope*freq)")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (rad)")
    plt.legend()

    return phase_corrected


def windowing(data_td):
    # ~14 ps pulse width
    dt = np.mean(np.diff(data_td[:, 0]))
    window_width = int(14 / dt)

    peak_pos = np.argmax(np.abs(data_td[:, 1]))
    window = signal.get_window("tukey", window_width)

    t_len = data_td.shape[0]
    window_start_idx = peak_pos - window_width // 2
    window = np.concatenate((np.zeros(window_start_idx),
                             window,
                             np.zeros(t_len - window_start_idx - len(window))))

    data_td_windowed = data_td.copy()
    data_td_windowed[:, 1] *= window

    plt.figure()
    plt.plot(data_td[:, 0], data_td[:, 1], label="Data w/o window")
    plt.plot(data_td_windowed[:, 0], data_td_windowed[:, 1], label="Data with window")
    plt.plot(data_td[:, 0], window * max(data_td[:, 1]), label="window * max(y)")
    plt.xlabel("Time (ps)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()

    return data_td_windowed
