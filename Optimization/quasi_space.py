import matplotlib.pyplot as plt

from imports import *
from Model.tmm_package import tmm_package_wrapper
from Measurements.measurements import select_measurements, get_avg_measurement
from Model.transmission_approximation import ri_approx
from functions import add_noise, do_fft
from Plotting.plot_data import plot_field


def main():
    d_list = [inf, 500, inf]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]

    refs, sams = select_measurements(keywords)

    avg_ref, avg_sam = get_avg_measurement(keywords)
    avg_ref_fd, avg_sam_fd = avg_ref.get_data_fd(reversed_time=True), avg_sam.get_data_fd(reversed_time=True)
    n = ri_approx(avg_ref_fd, avg_sam_fd, d_list[1])
    t = tmm_package_wrapper(d_list, n)

    mod_fd = array([avg_ref_fd[:, 0], t[:, 1] * avg_ref_fd[:, 1]]).T

    ref0_fd = refs[0].get_data_fd(reversed_time=True)
    sam0_fd = sams[0].get_data_fd(reversed_time=True)

    data_qs = do_fft(mod_fd)

    plt.figure()
    plt.plot(np.abs(data_qs[:, 1]))

    plot_field(mod_fd)

    plt.show()


if __name__ == '__main__':
    main()
