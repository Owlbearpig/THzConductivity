from imports import *
from numpy.fft import fft, ifft, fftfreq
from scipy import signal
from tmm import coh_tmm


def do_fft(t, y_td, pos_freqs_only=True):
    dt = float(np.mean(np.diff(t)))
    freqs, data_fd = fftfreq(n=len(t), d=dt), fft(y_td)

    if pos_freqs_only:
        post_freq_slice = freqs >= 0
        return freqs[post_freq_slice], data_fd[post_freq_slice]
    else:
        return freqs, data_fd


def do_ifft(freqs, y_fd):
    y_td = ifft(y_fd)
    t = np.arange(len(y_td)) / freqs.max()
    t += 1650

    return t, y_td


def phase_correction(data_fd, fit_range=None):
    if fit_range is None:
        fit_range = [0.25, 0.5]

    phase = np.angle(data_fd[:, 1])
    phase_unwrapped = np.unwrap(phase)

    fit_slice = (data_fd[:, 0] >= fit_range[0]) * (data_fd[:, 0] <= fit_range[1])
    p = np.polyfit(data_fd[fit_slice, 0], phase_unwrapped[fit_slice], 1)

    phase_corrected = phase_unwrapped - p[1].real

    if verbose:
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

def tmm_package_wrapper(freqs, d_list, n):
    # freq should be in THz ("between 0 and 10 THz"), d in um, n freq. resolved
    lam = (c0 / freqs) * 10 ** -6  # wl in um


    t_list = []
    for i, lambda_vac in enumerate(lam):
        n_list = [1, n[i], 1]
        t_list.append(coh_tmm("s", n_list, d_list, 0, lambda_vac)["t"])
    t_list = array(t_list)

    return t_list
