from Measurements.measurements import get_avg_measurement
from Plotting.plot_data import plot_ri, plot_field
from functions import phase_correction
from imports import *
from Model.tmm_package import tmm_package_wrapper

def ri_approx(ref_fd, sam_fd, thickness):
    # thickness in m, freq in THz
    freqs = ref_fd[:, 0].real
    omega = 2 * pi * freqs * THz

    fit_range = [0.25, 0.50]
    corrected_ref_phase = phase_correction(ref_fd, fit_range=fit_range)
    corrected_sam_phase = phase_correction(sam_fd, fit_range=fit_range)

    phase_diff = corrected_ref_phase - corrected_sam_phase

    if any(phase_diff[(fit_range[0] < freqs) * (freqs < fit_range[1])] < 0):
        phase_diff = -1 * phase_diff

    n = (1 + phase_diff * c0 / (thickness * omega)).real

    t_func = sam_fd[:, 1] / ref_fd[:, 1]

    k = - c0 * np.log(np.abs(t_func) * (n + 1) ** 2 / (4 * n)) / (omega * thickness)

    return array([freqs, n.real + 1j * k.real]).T


def no_fp_mod(ref_fd, n, thickness):
    freqs = ref_fd[:, 0].real
    omega = 2 * pi * freqs * THz

    t12, t21 = 2 / (1 + n[:, 1]), 2 * n[:, 1] / (n[:, 1] + 1)

    mod_fd = t12 * t21 * ref_fd[:, 1] * np.exp(1j * n[:, 1] * omega * thickness / c0)

    return array([freqs, mod_fd]).T


def main():
    keywords = ["GaAs", "Wafer", "25", "2021_08_24"]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]
    sam_thickess = 500 * um
    # keywords = ["InP 5", "2021_10_27"]
    # sam_thickess = 380 * um

    avg_ref_meas, avg_sam_meas = get_avg_measurement(keywords)
    avg_ref_fd, avg_sam_fd = avg_ref_meas.get_data_fd(), avg_sam_meas.get_data_fd()

    n = ri_approx(avg_ref_fd, avg_sam_fd, sam_thickess)

    plot_ri(n, label="RI, no FP original")

    d_list = [np.inf, sam_thickess / um, np.inf]
    t = tmm_package_wrapper(d_list, n)

    tmm_mod_fd = array([t[:, 0].real, t[:, 1] * avg_ref_fd[:, 1]]).T

    no_fp_mod_fd = no_fp_mod(avg_ref_fd, n, sam_thickess)

    pp_config = {"sub_offset": True, "en_windowing": False}
    avg_ref_meas, avg_sam_meas = get_avg_measurement(keywords, pp_config=pp_config)
    avg_ref_fd, avg_sam_fd = avg_ref_meas.get_data_fd(), avg_sam_meas.get_data_fd()

    plot_field(no_fp_mod_fd, label="No FP model", color="black")
    plot_field(tmm_mod_fd, label="tmm model", color="purple")
    plot_field(avg_ref_fd, label="Avg. Ref. ")
    plot_field(avg_sam_fd, label="Avg. Sam.")


if __name__ == '__main__':
    main()
    plt.show()
