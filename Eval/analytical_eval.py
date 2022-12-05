import matplotlib.pyplot as plt
from Model.tinkham_model import calc_sigma
from imports import *
from functions import pearson_corr_coeff
from Measurements.measurements import select_measurements
from Model.transmission_approximation import ri_approx
from Model.tmm_package import tmm_package_wrapper
from Plotting.plot_data import plot_ri, plot_field
from scipy.signal import correlate


def main():
    keywords_sub = ["GaAs_Wafer_25", "2021_08_09"]
    pp = {"sub_offset": True, "en_windowing": True}
    ref_sub_measurements, sam_sub_measurements = select_measurements(keywords_sub, post_process=pp)

    keywords_film = ["GaAs_C doped", "2021_08_09"]
    ref_film_measurements, sam_film_measurements = select_measurements(keywords_film, post_process=pp)

    d = [500 * 10 ** -6, 0.7 * 10 ** -6]

    p2p = []
    for meas_idx in range(10):
        sam_film_td = sam_film_measurements[meas_idx].get_data_td()
        p2p.append(np.abs(sam_film_td[:, 1].min()) + np.abs(sam_film_td[:, 1].max()))

    first_meas = sorted([sam_film_measurements[meas_idx] for meas_idx in range(10)], key=lambda x: x.meas_time)[0]
    t = [(sam_film_measurements[meas_idx].meas_time - first_meas.meas_time).total_seconds() / 60 for meas_idx in
         range(10)]
    plt.figure("P2p sam film")
    plt.title("P2p sam film")
    plt.plot(t, p2p)
    plt.xlabel("Time (min)")
    plt.ylabel("P2p")

    plot_field(first_meas.get_data_fd())

    for meas_idx in range(10):
        ref_film_fd = ref_film_measurements[meas_idx].get_data_fd()
        sam_film_fd = sam_film_measurements[meas_idx].get_data_fd()

        ref_sub_fd = ref_sub_measurements[meas_idx].get_data_fd()
        sam_sub_fd = sam_sub_measurements[meas_idx].get_data_fd()

        sigma = calc_sigma(ref_sub_fd, sam_sub_fd, ref_film_fd, sam_film_fd, d)

        freq_slice = (0.25 < ref_sub_fd[:, 0]) * (ref_sub_fd[:, 0] < 3.00)

        plt.figure("Sigma real")
        plt.title("Sigma real")
        plt.plot(ref_sub_fd[freq_slice, 0], sigma.real[freq_slice] / 100, label=f"Measurement index: {meas_idx}")
        plt.ylabel("Sigma real ($\Omega^{-1}$ $cm^{-1}$)")
        plt.xlabel("Frequency (THz)")
        plt.ylim((-10, 310))
        plt.legend()

        plt.figure("Sigma imag")
        plt.title("Sigma imag")
        plt.plot(ref_sub_fd[freq_slice, 0], sigma.imag[freq_slice] / 100, label=f"Measurement index: {meas_idx}")
        plt.ylabel("Sigma imag ($\Omega^{-1}$ $cm^{-1}$)")
        plt.xlabel("Frequency (THz)")
        # plt.ylim((-110, 110))
        plt.legend()


if __name__ == '__main__':
    main()

    plt.show()
