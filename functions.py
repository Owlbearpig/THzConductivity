import numpy as np

from imports import *
from numpy.fft import fft, ifft, fftfreq
from scipy import signal


def do_fft(t, y_td, pos_freqs_only=True):
    dt = float(np.mean(np.diff(t)))
    freqs, data_fd = fftfreq(n=len(t), d=dt), fft(y_td)

    if pos_freqs_only:
        post_freq_slice = freqs >= 0
        return freqs[post_freq_slice], data_fd[post_freq_slice]
    else:
        return freqs, data_fd


def do_ifft(data_fd, hermitian=False):
    freqs, y_fd = data_fd[:, 0], data_fd[:, 1]

    y_fd = nan_to_num(y_fd)

    if hermitian:
        y_fd = np.concatenate((y_fd, np.flip(np.conj(y_fd[1:]))))

    y_td = ifft(y_fd)
    t = np.arange(len(y_td)) / (2*freqs.max())
    t += 1650

    y_td = np.flip(y_td)

    return array([t, y_td]).T


def phase_correction(data_fd, freqs=None, fit_range=None):
    data_fd[:, 1] = nan_to_num(data_fd[:, 1])

    if fit_range is None:
        fit_range = [0.25, 0.5]

    if len(data_fd.shape) == 2:
        freqs = data_fd[:, 0]
        phase = np.angle(data_fd[:, 1])
    else:
        phase = np.angle(data_fd)

    phase_unwrapped = np.unwrap(phase)

    fit_slice = (freqs >= fit_range[0]) * (freqs <= fit_range[1])
    p = np.polyfit(freqs[fit_slice], phase_unwrapped[fit_slice], 1)

    phase_corrected = phase_unwrapped - p[1].real

    if verbose:
        plt.figure()
        plt.plot(freqs, phase_unwrapped, label="Unwrapped phase")
        plt.plot(freqs, phase_corrected, label="Shifted phase")
        plt.plot(freqs, freqs * p[0].real, label="Lin. fit (slope*freq)")
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
    # data_td_windowed[900:, 1] = data_td_windowed[900:, 1] * 0

    if verbose:
        plt.figure()
        plt.plot(data_td[:, 0], data_td[:, 1], label="Data w/o window")
        plt.plot(data_td_windowed[:, 0], data_td_windowed[:, 1], label="Data with window")
        plt.plot(data_td[:, 0], window * max(data_td[:, 1]), label="window * max(y)")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (a.u.)")
        plt.legend()

    return data_td_windowed



def calc_absorption(freqs, k):
    # Assuming freqs in range (0, 10 THz), returns a in units of 1/cm (1/m * 1/100)
    omega = 2 * pi * freqs * THz
    a = (2 * omega * k) / c0

    return a / 100
