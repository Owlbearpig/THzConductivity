import matplotlib.pyplot as plt

from imports import *
from functions import do_ifft, phase_correction, cauchy_relation
from Measurements.measurements import get_avg_measurement
from Model.transmission_approximation import ri_approx
from functools import partial
from Model.tmm_package import tmm_package_wrapper
from helpers import get_closest_idx
from Plotting.plot_data import plot
from scipy.signal import correlate
from scipy.stats import pearsonr


class Cost:
    def __init__(self, d_list, keywords, simulated_sample=False, local_verbose=False):
        self.keywords = keywords
        self.d_list = d_list
        self.simulated_sample = simulated_sample
        self.verbose = local_verbose

        self.n_approx = self.ri_approximation()

        self.ref_data_td, self.sam_data_td = None, None
        self.freqs = None
        self.sam_phase_unwrapped, self.ref_phase_unwrapped = None, None
        self.ref_data_fd, self.sam_data_fd = self.eval_measurement()

    def eval_measurement(self):
        pp_config = {"sub_offset": True, "en_windowing": False}
        avg_ref, avg_sam = get_avg_measurement(self.keywords, pp_config=pp_config)

        self.ref_data_td, self.sam_data_td = avg_ref.get_data_td(), avg_sam.get_data_td()

        ref_fd = avg_ref.get_data_fd(reversed_time=True)
        sam_fd = avg_sam.get_data_fd(reversed_time=True)

        self.freqs = ref_fd[:, 0].real

        if self.simulated_sample:
            # use n_approx to simulate sample; sam_sim = ref_measured * t_model
            t = tmm_package_wrapper(ref_fd[:, 0], self.d_list, self.n_approx)

            sam_fd[:, 1] = ref_fd[:, 1] * t
            self.sam_data_td = do_ifft(sam_fd, hermitian=True)

        self.sam_phase_unwrapped = phase_correction(sam_fd)
        self.ref_phase_unwrapped = phase_correction(ref_fd)

        return ref_fd, sam_fd

    def ri_approximation(self):
        pp_config = {"sub_offset": True, "en_windowing": True}
        avg_ref, avg_sam = get_avg_measurement(self.keywords, pp_config=pp_config)

        n_approx = ri_approx(avg_ref.get_data_fd(), avg_sam.get_data_fd(), self.d_list[1] * um)

        return n_approx

    def cost(self, freq_idx, p, *args):
        if isinstance(freq_idx, float):
            freq_idx = get_closest_idx(self.freqs, freq_idx)

        n = p[0] + 1j * p[1]

        t = tmm_package_wrapper(self.freqs[freq_idx], self.d_list, n)

        y_fd_mod = t * self.ref_data_fd[freq_idx, 1]

        amp_loss = (y_fd_mod.real - self.sam_data_fd[freq_idx, 1].real) ** 2
        phase_loss = (np.angle(y_fd_mod) - np.angle(self.sam_data_fd[freq_idx, 1])) ** 2
        # phase_loss = (y_fd_mod.imag - self.sam_data_fd[freq_idx, 1].imag) ** 2

        loss = amp_loss + phase_loss
        # print(amp_loss, phase_loss)
        loss = np.log10(loss)

        return loss

    def td_cost(self, freq_index, p):
        n = self.n_approx.copy()

        n[freq_index] = p[0] + 1j * p[1]

        t = tmm_package_wrapper(self.freqs, self.d_list, n)

        y_fd_sam_mod = t * self.ref_data_fd[:, 1]

        td_sam_mod = do_ifft(y_fd_sam_mod, hermitian=True)

        # pears = pearsonr(self.sam_data_td[:, 1], y_td_sam_mod)

        loss = np.sum((np.abs(td_sam_mod[:, 1]) - np.abs(self.sam_data_td[:, 1])) ** 2) / len(td_sam_mod[:, 1])

        plt.figure()
        plt.title("Time domain")
        plt.plot(td_sam_mod[:, 0], td_sam_mod[:, 1], label="Sam. model", color="black")
        plt.plot(self.ref_data_td[:, 0], self.ref_data_td[:, 1], label="Ref. measurement")
        plt.plot(self.sam_data_td[:, 0], self.sam_data_td[:, 1], label="Sam. measurement")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (a.u.)")
        plt.legend()

        return loss

    def cost_1d(self, freq_idx, p):
        if isinstance(freq_idx, float):
            freq_idx = get_closest_idx(self.freqs, freq_idx)

        n = p[0] + 1j * self.n_approx[freq_idx].imag

        t = tmm_package_wrapper(self.freqs[freq_idx], self.d_list, n)

        y_fd_mod = t * self.ref_data_fd[freq_idx, 1]

        amp_loss = 0 * (y_fd_mod.real - self.sam_data_fd[freq_idx, 1].real) ** 2
        phase_loss = (np.angle(y_fd_mod) - np.angle(self.sam_data_fd[freq_idx, 1])) ** 2
        phase_loss = (y_fd_mod.imag - self.sam_data_fd[freq_idx, 1].imag) ** 2

        loss = amp_loss + phase_loss
        # print(amp_loss, phase_loss)
        # loss = np.log10(loss)

        return loss

    def cost_unwrapped_phase(self, freq_idx, p):
        if isinstance(freq_idx, float):
            freq_idx = get_closest_idx(self.freqs, freq_idx)

        n = self.n_approx.copy()
        n[freq_idx] = p[0] + 1j * self.n_approx[freq_idx].imag

        t = tmm_package_wrapper(self.freqs, self.d_list, n)

        mod_fd = array([self.ref_data_fd[:, 0], t * self.ref_data_fd[:, 1]]).T

        mod_phase_unwrapped = phase_correction(mod_fd)

        eval_range = get_closest_idx(self.freqs, 0.05), get_closest_idx(self.freqs, 3.00)

        loss = np.sum((mod_phase_unwrapped[eval_range[0]:eval_range[1]] -
                       self.sam_phase_unwrapped[eval_range[0]:eval_range[1]]) ** 2)

        return loss

    def cost_cauchy_relation(self, p, en_plot = False):
        n = cauchy_relation(self.freqs, p)

        t = tmm_package_wrapper(self.freqs, self.d_list, n)

        mod_fd = array([self.ref_data_fd[:, 0], t * self.ref_data_fd[:, 1]]).T

        if en_plot:
            plt.figure("Refractive index, cauchy relation")
            plt.title("Refractive index, cauchy relation")
            plt.plot(self.freqs, n)

            plot(mod_fd, label="model")
            plot(self.sam_data_fd, label="sam")

        mod_phase_unwrapped = phase_correction(mod_fd)

        eval_range = get_closest_idx(self.freqs, 0.20), get_closest_idx(self.freqs, 3.00)

        loss = np.sum((mod_phase_unwrapped[eval_range[0]:eval_range[1]] -
                       self.sam_phase_unwrapped[eval_range[0]:eval_range[1]]) ** 2)

        return loss


if __name__ == '__main__':
    d_list = [inf, 500, inf]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]

    freq = 0.450

    new_cost = Cost(d_list, keywords, simulated_sample=False, local_verbose=True)

    cost_func = partial(new_cost.cost_cauchy_relation)
    freq_idx = get_closest_idx(new_cost.freqs, freq)


    a = 3.20
    b_line = np.linspace(0.000254, 0.000354, 100)
    cost_vals = []
    for i, b in enumerate(b_line):
        print(i, b)
        #cost_vals.append(cost_func([3.658, 0.000354]))
        cost_vals.append(cost_func([a, b]))

    plt.figure()
    plt.plot(cost_vals)

    b = b_line[np.argmin(cost_vals)]
    print(b)
    print(min(cost_vals))
    new_cost.cost_cauchy_relation(p=[a, b], en_plot=True)

    plt.show()
