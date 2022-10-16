from imports import *
from functions import do_ifft
from Measurements.measurements import get_avg_measurement
from Model.transmission_approximation import ri_approx
from functools import partial
from Model.tmm_package import tmm_package_wrapper
from helpers import get_closest_idx
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
        #phase_loss = (y_fd_mod.imag - self.sam_data_fd[freq_idx, 1].imag) ** 2

        loss = amp_loss + phase_loss
        #print(amp_loss, phase_loss)
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


if __name__ == '__main__':
    d_list = [inf, 500, inf]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]

    freq = 0.3099

    new_cost = Cost(d_list, keywords, simulated_sample=False, local_verbose=True)
    cost_func = partial(new_cost.cost, freq)
    freq_idx = get_closest_idx(new_cost.freqs, freq)
    for i in range(0, 10):
        val = new_cost.sam_data_fd[freq_idx + 5 - i]
        print(val)
        print(np.abs(val[1]), np.angle(val[1]), "\n")
    n_goal = new_cost.n_approx[freq_idx]

    bounds = (0.0001, 0.2)
    rez_y = 500
    k_line = np.linspace(bounds[0], bounds[1], rez_y)

    cost_vals = []
    for k in k_line:
        cost_vals.append(cost_func([n_goal.real * 1, k]))

    plt.figure()
    plt.plot(k_line, cost_vals, label="cost value")
    plt.title("Value of cost as a function of k,\n at goal refractive index")
    plt.legend()
    plt.ylabel("Value of cost function")
    plt.xlabel("k")

    print(f"Goal k: {n_goal.imag}")

    plt.show()
