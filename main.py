import matplotlib.pyplot as plt

from Measurements.measurements import select_measurements, get_avg_measurement
from imports import *
from functions import do_ifft, add_noise
from Model.tmm_package import tmm_package_wrapper
from Model.transmission_approximation import ri_approx
from Plotting.plot_data import plot


def main():
    d_list = [inf, 500, inf]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]

    refs, sams = select_measurements(keywords)

    avg_ref, avg_sam = get_avg_measurement(keywords)
    avg_ref_fd, avg_sam_fd = avg_ref.get_data_fd(reversed_time=True), avg_sam.get_data_fd(reversed_time=True)
    n = ri_approx(avg_ref_fd, avg_sam_fd, d_list[1] * um)
    t = tmm_package_wrapper(avg_ref_fd[:, 0], d_list, n)

    mod_fd = array([avg_ref_fd[:, 0], t * avg_ref_fd[:, 1]]).T

    ref0_fd = refs[0].get_data_fd(reversed_time=True)
    sam0_fd = sams[0].get_data_fd(reversed_time=True)

    mod_fd_noise = add_noise(mod_fd, en_plots=True, seed=420)

    plot(mod_fd, label="model")
    # plot(avg_ref_fd, label="ref")
    # plot(avg_sam_fd, label="sample0")
    plot(mod_fd_noise, label="mod with noise")

    plt.show()


if __name__ == '__main__':
    main()
    plt.show()
