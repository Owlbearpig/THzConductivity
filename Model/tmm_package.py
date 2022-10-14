from imports import *
from Measurements.measurements import get_all_measurements
from helpers import select_measurements
from functions import do_ifft, tmm_package_wrapper


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

    ref_data_td, sam_data_td = refs[0].get_data_td(), sams[0].get_data_td()
    t_td_ref, y_td_ref = ref_data_td[:, 0], ref_data_td[:, 1]
    t_td_sam, y_td_sam = sam_data_td[:, 0], sam_data_td[:, 1]

    ref_data_fd, sam_data_fd = refs[0].get_data_fd(reversed_time=True), sams[0].get_data_fd(reversed_time=True)
    freqs, y_fd_ref = ref_data_fd[:, 0], ref_data_fd[:, 1]
    freqs, y_fd_sam = sam_data_fd[:, 0], sam_data_fd[:, 1]

    y_fd_sam_mod = t * y_fd_ref
    y_fd_sam_mod = nan_to_num(y_fd_sam_mod)
    t_td_sam_mod, y_td_sam_mod = do_ifft(freqs, y_fd_sam_mod)
    y_td_sam_mod = np.flip(y_td_sam_mod)

    plt.figure()
    plt.plot(freqs, np.abs(y_fd_sam_mod), label="Sam. model", color="black")
    plt.plot(freqs, np.abs(y_fd_ref), label="Ref. measurement")
    plt.plot(freqs, np.abs(y_fd_sam), label="Sam. measurement")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()

    plt.figure()
    plt.plot(t_td_sam_mod, y_td_sam_mod, label="Sam. model", color="black")
    plt.plot(t_td_ref, y_td_ref, label="Ref. measurement")
    plt.plot(t_td_sam, y_td_sam, label="Sam. measurement")
    plt.xlabel("Time (ps)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()


if __name__ == '__main__':
    main()
    plt.show()