from imports import *
from Measurements.measurements import get_all_measurements, select_measurements
from Plotting.plot_data import plot
from helpers import is_iterable, get_closest_idx
from tmm import coh_tmm


def tmm_package_wrapper(freqs, d_list, n):
    # freq should be in THz ("between 0 and 10 THz"), d in um
    lam = (c0 / freqs) * 10 ** -6  # wl in um

    if (not is_iterable(n)) or (not is_iterable(freqs)):
        n_list = [1, n, 1]
        t_list = array([coh_tmm("s", n_list, d_list, 0, lam)["t"]])
    else:
        t_list = []
        for i, lambda_vac in enumerate(lam):
            n_list = [1, n[i], 1]
            t_list.append(coh_tmm("s", n_list, d_list, 0, lambda_vac)["t"])
        t_list = array(t_list)

    return t_list


def main():
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]

    freqs = np.load("freqs_" + "_".join(keywords) + ".npy")
    n_real = np.load("n_" + "_".join(keywords) + ".npy")
    n_imag = np.load("k_" + "_".join(keywords) + ".npy")

    # freq_slice = (freqs >= 0.25) * (freqs <= 3.00)
    freq_slice = (freqs >= 0.00) * (freqs <= 10.00)

    n_real = n_real[freq_slice]
    n_imag = n_imag[freq_slice]

    freqs = freqs[freq_slice]

    d_list = [inf, 500, inf]
    n = n_real + 1j * n_imag

    #n[get_closest_idx(freqs, 0.070)] = 3.00 + 1j*n[get_closest_idx(freqs, 0.070)].imag

    t = tmm_package_wrapper(freqs, d_list, n)

    plt.figure()
    plt.plot(freqs, np.abs(t) ** 2)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Fraction of power transmitted")
    plt.title("Transmission at normal incidence")

    measurements = get_all_measurements()

    selected_measurements = select_measurements(measurements, keywords)

    refs = [x for x in selected_measurements if x.meas_type == "ref"]
    sams = [x for x in selected_measurements if x.meas_type == "sam"]

    ref_fd, sam_fd = refs[0].get_data_fd(reversed_time=True), sams[0].get_data_fd(reversed_time=True)

    mod_fd = array([freqs, t * ref_fd[:, 1]]).T

    plot(ref_fd, label="Ref. measurement")
    plot(mod_fd, label="Sam. model")
    plot(sam_fd, label="Sam. measurement")
    plt.show()


if __name__ == '__main__':
    main()
    plt.show()
