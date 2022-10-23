import numpy as np

from imports import *
from Measurements.measurements import get_all_measurements, select_measurements
from Plotting.plot_data import plot_field, plot_ri
from helpers import is_iterable, get_closest_idx
from tmm import coh_tmm
from Results.parse_teralyzer_results import select_results
from scipy.interpolate import interp1d


def tmm_package_wrapper(d_list, n, add_air_phase=True):
    # freq should be in THz ("between 0 and 10 THz"), d in um (wl in um)
    if n.ndim == 1:
        freqs = n[0].real
        lam = (c0 / freqs) * 10 ** -6
        n_list = [1, n[1], 1]
        t_list = array([coh_tmm("s", n_list, d_list, 0, lam)["t"]])
    else:
        freqs = n[:, 0].real
        lam = (c0 / freqs) * 10 ** -6

        t_list = []
        for i, lambda_vac in enumerate(lam):
            n_list = [1, n[i, 1], 1]
            t_list.append(coh_tmm("s", n_list, d_list, 0, lambda_vac)["t"])
        t_list = array(t_list)

    t_list = array(t_list)
    if add_air_phase:
        omega = 2 * pi * freqs * THz
        d = d_list[1] * um
        t_list *= np.exp(-1j * omega * d / c0)

    return array([freqs, t_list]).T


def tmm_from_ri(n, d_list, ref_fd, en_plot=False):
    freqs = ref_fd[:, 0].real
    freq_slice = (freqs >= n[:, 0].real.min()) * (freqs <= n[:, 0].real.max())
    freqs = freqs[freq_slice]

    n_interpolator = interp1d(n[:, 0].real, n[:, 1], kind="linear")
    n_interp = array([freqs, n_interpolator(freqs)]).T

    if en_plot:
        plot_ri(n, label="Original data")
        plot_ri(n_interp, label="Interpolated data")

    t = tmm_package_wrapper(d_list, n_interp)
    mod_fd = t[:, 1] * ref_fd[freq_slice, 1]

    df = np.mean(np.diff(freqs))
    min_freq, max_freq = freqs.min(), freqs.max()
    freqs = np.concatenate((np.arange(0, min_freq, df), freqs, np.arange(max_freq, 10, df)))

    leading_0, trailing_0 = np.zeros(len(np.arange(0, min_freq, df))), np.zeros(len(np.arange(max_freq, 10, df)))
    mod_fd = np.concatenate((leading_0, mod_fd, trailing_0))

    return array([freqs, mod_fd]).T


def tmm_teralyzer_result(keywords, d_list, ref_fd):
    # use refractive index from teralyzer to calculate t at ref freqs.
    #   -> mod_fd = t * ref_fd

    res = select_results(keywords)[0]
    print("Using refractive index of result: ", res)
    n = res.get_n()

    mod_teralyzer = tmm_from_ri(n, d_list, ref_fd, en_plot=False)

    return mod_teralyzer


def main():
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]

    freqs = np.load("freqs_" + "_".join(keywords) + ".npy")
    n_real = np.load("n_" + "_".join(keywords) + ".npy")
    n_imag = np.load("k_" + "_".join(keywords) + ".npy")

    freq_slice = (freqs >= 0.00) * (freqs <= 10.00)

    n_real = n_real[freq_slice]
    n_imag = n_imag[freq_slice]

    freqs = freqs[freq_slice]

    d_list = [inf, 500, inf]

    n = array([freqs, n_real + 1j * n_imag]).T

    t = tmm_package_wrapper(d_list, n)

    plt.figure()
    plt.plot(freqs, np.abs(t[:, 1]) ** 2)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Fraction of power transmitted")
    plt.title("Transmission at normal incidence")

    measurements = get_all_measurements()

    refs, sams = select_measurements(measurements, keywords)

    ref_fd, sam_fd = refs[0].get_data_fd(), sams[0].get_data_fd()

    mod_fd = array([freqs, t[:, 1] * ref_fd[:, 1]]).T

    plot_field(ref_fd, label="Ref. measurement")
    plot_field(mod_fd, label="Sam. model")
    plot_field(sam_fd, label="Sam. measurement")


if __name__ == '__main__':
    main()
    plt.show()
