import matplotlib.pyplot as plt

from Measurements.measurements import select_measurements, get_avg_measurement
from imports import *
from functions import add_noise
from Model.tmm_package import tmm_package_wrapper, tmm_teralyzer_result
from Model.transmission_approximation import ri_approx
from Plotting.plot_data import plot_field, plot_ri


def main():
    d_list = [inf, 500, inf]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]

    refs, sams = select_measurements(keywords)

    pp_config = {"sub_offset": True, "en_windowing": True}
    avg_ref, avg_sam = get_avg_measurement(keywords, pp_config=pp_config)
    avg_ref_fd, avg_sam_fd = avg_ref.get_data_fd(), avg_sam.get_data_fd()
    n = ri_approx(avg_ref_fd, avg_sam_fd, d_list[1] * um)
    t = tmm_package_wrapper(d_list, n)

    plot_ri(n, label="Ri approximation")

    pp_config = {"sub_offset": True, "en_windowing": False}
    avg_ref, avg_sam = get_avg_measurement(keywords, pp_config=pp_config)
    avg_ref_fd, avg_sam_fd = avg_ref.get_data_fd(), avg_sam.get_data_fd()

    mod_fd = array([avg_ref_fd[:, 0], t[:, 1] * avg_ref_fd[:, 1]]).T

    ref0_fd = refs[0].get_data_fd()
    sam0_fd = sams[0].get_data_fd()

    plot_field(mod_fd, label="model", color="black")
    plot_field(avg_ref_fd, label="ref")
    plot_field(avg_sam_fd, label="sample0")

    # mod_fd_noise = add_noise(mod_fd, en_plots=True, seed=420)
    # plot(mod_fd_noise, label="mod with noise")

    mod_fd_teralyzer = tmm_teralyzer_result([keywords[-1], "wider500"], d_list, avg_ref_fd, en_plot=True)

    plot_field(mod_fd_teralyzer, label="model teralyzer", color="red")


if __name__ == '__main__':
    main()
    plt.show()
