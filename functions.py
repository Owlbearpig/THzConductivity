import matplotlib.pyplot as plt
import numpy as np

from imports import *
from numpy.fft import fft, ifft, fftfreq
from scipy import signal


def do_fft(data_td, pos_freqs_only=True):
    data_td = nan_to_num(data_td)

    dt = float(np.mean(np.diff(data_td[:, 0])))
    freqs, data_fd = fftfreq(n=len(data_td[:, 0]), d=dt), fft(data_td[:, 1])

    if pos_freqs_only:
        post_freq_slice = freqs >= 0
        return array([freqs[post_freq_slice], data_fd[post_freq_slice]]).T
    else:
        return array([freqs, data_fd]).T


def do_ifft(data_fd, hermitian=False):
    freqs, y_fd = data_fd[:, 0], data_fd[:, 1]

    y_fd = nan_to_num(y_fd)

    if hermitian:
        y_fd = np.concatenate((y_fd, np.flip(np.conj(y_fd[1:]))))

    y_td = ifft(y_fd)
    t = np.arange(len(y_td)) / (1 * freqs.max())
    t += 1650

    y_td = np.flip(y_td)

    return array([t, y_td]).T


def unwrap(data_fd):
    if data_fd.ndim == 2:
        y = nan_to_num(data_fd[:, 1])
    else:
        y = nan_to_num(data_fd)

    phase = np.angle(y)

    phase_unwrapped = np.unwrap(phase)

    return np.abs(phase_unwrapped)


def phase_correction(data_fd, fit_range=None, verbose=verbose):
    freqs = data_fd[:, 0]

    phase_unwrapped = unwrap(data_fd)

    if fit_range is None:
        fit_range = [0.25, 0.50]

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


def cauchy_relation(freqs, p):
    lam = (c0 / freqs) * 10 ** -9

    n = np.zeros_like(lam)
    for i, coeff in enumerate(p):
        n += coeff * lam ** (-2 * i)

    return n


def add_noise(data_fd, enabled=True, scale=0.05, seed=None, en_plots=False):
    data_ret = nan_to_num(data_fd)

    np.random.seed(seed)

    if not enabled:
        return data_ret

    noise_phase = np.random.normal(0, scale*2, len(data_fd[:, 0]))
    noise_amp = np.random.normal(0, scale*1.5, len(data_fd[:, 0]))

    phi, magn = np.angle(data_fd[:, 1]), np.abs(data_fd[:, 1])

    phi_noisy = phi + noise_phase
    magn_noisy = magn * (1 + noise_amp)

    if en_plots:
        freqs = data_ret[:, 0]

        plt.figure("Phase")
        plt.plot(freqs, phi, label="Original data")
        plt.plot(freqs, phi_noisy, label="+ noise")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")
        plt.legend()

        plt.figure("Spectrum")
        plt.plot(freqs, magn, label="Original data")
        plt.plot(freqs, magn_noisy, label="+ noise")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (a.u.)")
        plt.legend()
        plt.show()

    noisy_data = magn_noisy * np.exp(1j*phi_noisy)

    data_ret[:, 1] = noisy_data.real + 1j * noisy_data.imag

    return data_ret
