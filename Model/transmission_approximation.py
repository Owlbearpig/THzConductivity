import matplotlib.pyplot as plt

from imports import *
from functions import phase_correction, calc_absorption
from Measurements.measurements import get_avg_measurement


def ri_approx(ref_data_fd, sam_data_fd, thickness):
    # thickness in m, freq in THz
    freqs = ref_data_fd[:, 0]
    omega = 2 * pi * freqs * THz

    fit_range = [0.25, 0.5]
    corrected_ref_phase = phase_correction(ref_data_fd, fit_range=fit_range)
    corrected_sam_phase = phase_correction(sam_data_fd, fit_range=fit_range)

    phase_diff = corrected_ref_phase - corrected_sam_phase
    if any(phase_diff[(fit_range[0] < freqs)*(freqs < fit_range[1])] < 0):
        phase_diff = -1 * phase_diff

    n = (1 + phase_diff * c0 / (thickness * omega)).real

    t_func = sam_data_fd[:, 1] / ref_data_fd[:, 1]

    k = - c0 * np.log(np.abs(t_func) * (n + 1) ** 2 / (4 * n)) / (omega * thickness)

    return n.real + 1j*k.real


def main():
    keywords = ["GaAs", "Wafer", "25", "2021_08_24"]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]
    sam_thickess = 500 * um
    # keywords = ["InP 5", "2021_10_27"]
    # sam_thickess = 380 * um

    pp_config = {"sub_offset": True, "en_windowing": True}
    avg_ref, avg_sam = get_avg_measurement(keywords, pp_config=pp_config)

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
    plt.title("Extinction coefficient")
    plt.plot(freqs[(freqs > 0.00) * (freqs < 3.00)], k[(freqs > 0.00) * (freqs < 3.00)])
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Extinction coefficient")

    plt.figure()
    plt.title("Refractive index")
    plt.plot(freqs[(freqs > 0.25) * (freqs < 3.00)], n[(freqs > 0.25) * (freqs < 3.00)])
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Refractive index")

    plt.figure()
    plt.title("Absorption coefficient")
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
