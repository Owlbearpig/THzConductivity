from imports import *
from functions import phase_correction
from Measurements.measurements import Measurement, get_all_measurements, avg_data
from helpers import select_measurements


def ri_approx(ref_data_fd, sam_data_fd, thickness):
    omega = 2 * pi * ref_data_fd[:, 0] * THz

    corrected_ref_phase = phase_correction(ref_data_fd)
    corrected_sam_phase = phase_correction(sam_data_fd)

    phase_diff = corrected_ref_phase - corrected_sam_phase

    n = 1 + phase_diff * c0 / (thickness * omega)

    t_func = sam_data_fd[:, 1] / ref_data_fd[:, 1]

    k = - c0 * np.log(np.abs(t_func) * (n + 1) ** 2 / (4 * n)) / (omega * thickness)
    a = (2 * omega * k) / c0

    return n, a / 100, k


def main():
    pp_config = {"sub_offset": True, "en_windowing": True}
    all_measurements = get_all_measurements()
    keywords = ["GaAs", "Wafer", "25", "2021_08_24"]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]
    sam_thickess = 500 * um
    # keywords = ["InP 5", "2021_10_27"]
    # sam_thickess = 380 * um

    selected_measurements = select_measurements(all_measurements, keywords)

    refs = [measurement for measurement in selected_measurements if measurement.meas_type == "ref"]
    sams = [measurement for measurement in selected_measurements if measurement.meas_type == "sam"]

    avg_ref = Measurement(data_td=avg_data(refs), meas_type="ref", post_process_config=pp_config)
    avg_sam = Measurement(data_td=avg_data(sams), meas_type="sam", post_process_config=pp_config)

    avg_ref_data_td, avg_sam_data_td = avg_ref.get_data_td(), avg_sam.get_data_td()
    avg_ref_data_fd, avg_sam_data_fd = avg_ref.get_data_fd(), avg_sam.get_data_fd()

    if verbose:
        plt.figure()
        plt.plot(avg_ref_data_td[:, 0], avg_ref_data_td[:, 1])
        plt.plot(avg_sam_data_td[:, 0], avg_sam_data_td[:, 1])

    freqs = avg_ref_data_fd[:, 0].real

    if verbose:
        plt.figure()
        plt.plot(freqs, np.log10(np.abs(avg_ref_data_fd[:, 1])))
        plt.plot(freqs, np.log10(np.abs(avg_sam_data_fd[:, 1])))

    n, a, k = ri_approx(avg_ref_data_fd, avg_sam_data_fd, sam_thickess)

    plt.figure()
    plt.plot(freqs[(freqs > 0.25) * (freqs < 3.00)], n[(freqs > 0.25) * (freqs < 3.00)])
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Refractive index")

    plt.figure()
    plt.plot(freqs[(freqs > 0.25) * (freqs < 3.00)], a[(freqs > 0.25) * (freqs < 3.00)])
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Absorption coefficient (1/cm)")

    np.save("freqs_" + "_".join(keywords), freqs)
    np.save("n_" + "_".join(keywords), n)
    np.save("a_" + "_".join(keywords), a)
    np.save("k_" + "_".join(keywords), k)

if __name__ == '__main__':
    main()
    plt.show()
