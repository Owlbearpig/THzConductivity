import matplotlib.pyplot as plt

from imports import *
from functions import pearson_corr_coeff
from Measurements.measurements import select_measurements
from Model.transmission_approximation import ri_approx
from Model.tmm_package import tmm_package_wrapper
from Plotting.plot_data import plot_ri, plot_field
from scipy.signal import correlate


def main():
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]
    d_list = [inf, 500, inf]
    meas_idx = 3

    pp = {"sub_offset": True, "en_windowing": True}
    ref_measurements_noFP, sam_measurements_noFP = select_measurements(keywords, post_process=pp)
    ref_fd_noFP = ref_measurements_noFP[meas_idx].get_data_fd()
    sam_fd_noFP = sam_measurements_noFP[meas_idx].get_data_fd()

    n = ri_approx(ref_fd_noFP, sam_fd_noFP, d_list[1])

    plot_ri(n, label=f"measurement {meas_idx}")

    pp = {"sub_offset": True, "en_windowing": False}
    ref_measurements, sam_measurements = select_measurements(keywords, post_process=pp)
    ref_fd = ref_measurements[meas_idx].get_data_fd()
    sam_fd = sam_measurements[meas_idx].get_data_fd()

    t = tmm_package_wrapper(d_list, n)

    tmm_mod_fd = array([t[:, 0].real, t[:, 1] * ref_fd[:, 1]]).T

    freq_idx_slice = slice(50, 700)
    ref_fd = ref_fd[freq_idx_slice, :]
    sam_fd = sam_fd[freq_idx_slice, :]
    tmm_mod_fd = tmm_mod_fd[freq_idx_slice, :]

    plot_field(tmm_mod_fd, label="TMM(n_analytical)", color="black")
    plot_field(ref_fd, label="Avg. Ref. ")
    plot_field(sam_fd, label="Avg. Sam.")

    corr = pearson_corr_coeff(tmm_mod_fd, sam_fd)

    print(meas_idx, corr)


if __name__ == '__main__':
    main()
    plt.show()
