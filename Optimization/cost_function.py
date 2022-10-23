from imports import *
from functions import do_ifft, phase_correction, add_noise
from Measurements.measurements import get_avg_measurement, select_measurements
from Model.transmission_approximation import ri_approx
from functools import partial
from Model.tmm_package import tmm_package_wrapper
from helpers import get_closest_idx
from scipy.optimize import shgo, minimize, basinhopping


class Cost:
    def __init__(self, d_list, keywords, sam_idx=None, simulated_sample=False, en_noise=False):
        self.keywords = keywords
        self.d_list = d_list
        self.simulated_sample = simulated_sample
        self.en_noise = en_noise
        self.sam_idx = sam_idx

        self.n_approx = self.ri_approximation()

        self.ref_data_td, self.sam_data_td = None, None
        self.freqs = None
        self.sam_phase_unwrapped, self.ref_phase_unwrapped = None, None
        self.ref_data_fd, self.sam_data_fd = self.eval_measurement()

    def eval_measurement(self):
        if self.sam_idx is not None:
            refs, sams = select_measurements(self.keywords)
            ref, sam = refs[self.sam_idx], sams[self.sam_idx]
        else:
            pp_config = {"sub_offset": True, "en_windowing": False}
            ref, sam = get_avg_measurement(self.keywords, pp_config=pp_config)

        self.ref_data_td, self.sam_data_td = ref.get_data_td(), sam.get_data_td()

        ref_fd = ref.get_data_fd()
        sam_fd = sam.get_data_fd()

        self.freqs = ref_fd[:, 0].real

        if self.simulated_sample:
            # use n_approx to simulate sample; sam_sim = ref_measured * t_model
            t = tmm_package_wrapper(self.d_list, self.n_approx)

            sam_fd[:, 1] = ref_fd[:, 1] * t[:, 1]
            self.sam_data_td = do_ifft(sam_fd)

        if self.en_noise:
            sam_fd = add_noise(sam_fd, scale=0.005)

        self.sam_phase_unwrapped = phase_correction(sam_fd)
        self.ref_phase_unwrapped = phase_correction(ref_fd)

        return ref_fd, sam_fd

    def ri_approximation(self):
        avg_ref, avg_sam = get_avg_measurement(self.keywords)

        n_approx = ri_approx(avg_ref.get_data_fd(), avg_sam.get_data_fd(), self.d_list[1])

        return n_approx

    def cost(self, freq_idx, p, *args):
        if isinstance(freq_idx, float):
            freq_idx = get_closest_idx(self.freqs, freq_idx)

        n = array([self.freqs[freq_idx], p[0] + 1j * p[1]]).T

        t = tmm_package_wrapper(self.d_list, n)

        y_fd_mod = t[1] * self.ref_data_fd[freq_idx, 1]

        amp_loss = np.abs(y_fd_mod) - np.abs(self.sam_data_fd[freq_idx, 1])
        phase_loss = np.angle(y_fd_mod) - np.angle(self.sam_data_fd[freq_idx, 1])
        # phase_loss = (y_fd_mod.imag - self.sam_data_fd[freq_idx, 1].imag) ** 2

        loss = np.abs(amp_loss) + np.abs(phase_loss)
        # print(amp_loss, phase_loss)
        #loss = np.log10(loss)

        return loss

    def cost_range(self, freq_idx, width, p):
        width = width
        freqs = self.ref_data_fd[:, 0]
        freq_range = freqs[freq_idx:freq_idx + width]

        n = array([freq_range, array([p[2 * i] + 1j * p[2 * i + 1] for i in range(width)])]).T
        t = tmm_package_wrapper(self.d_list, n)

        y_fd_mod = t[:, 1] * self.ref_data_fd[freq_idx:freq_idx + width, 1]

        amp_loss = np.sum(np.abs(y_fd_mod) - np.abs(self.sam_data_fd[freq_idx:freq_idx + width, 1]))

        phase_loss = np.sum(np.angle(y_fd_mod) - np.angle(self.sam_data_fd[freq_idx:freq_idx + width, 1]))

        penalty = np.sum(np.diff(n.real) ** 2 / width)
        # print(amp_loss, phase_loss)
        # print(np.log10(amp_loss), np.log10(phase_loss))
        # return np.log10(amp_loss) + np.log10(phase_loss) #+ np.log10(penalty)
        return np.abs(amp_loss) + np.abs(phase_loss)


if __name__ == '__main__':
    d_list = [inf, 500, inf]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]

    freq = 0.605

    new_cost = Cost(d_list, keywords, sam_idx=0)
    freq_idx = get_closest_idx(new_cost.freqs, freq)

    cost_func = partial(new_cost.cost, freq_idx)
    p_goal = new_cost.n_approx[freq_idx]

    p0 = 1 * p_goal
    p0 = p0.real + 1j * p0.imag
    print(p0)
    print(cost_func([p0.real, p0.imag]))

    bounds = [[3.4, 3.9], [0.002, 0.020]]
    # bounds = width * [[3.4, 4.0], [0.002, 0.020]]
    print(bounds)

    # minimizer_kwargs = {"tol": 1e-14, "method": "Nelder-Mead", "bounds": bounds}
    minimizer_kwargs = {"bounds": bounds, "method": "Nelder-Mead"}
    # res = shgo(cost_func, bounds=bounds, n=100, iters=2, minimizer_kwargs=minimizer_kwargs, options={"disp": True})
    # res = shgo(cost_func, bounds=bounds, options={"disp": True}, minimizer_kwargs = {"bounds": bounds})

    # bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
    # res = basinhopping(cost_func, x0=p0, stepsize=0.02, niter=100, minimizer_kwargs=minimizer_kwargs, take_step=bounded_step)

    res = minimize(cost_func, x0=array([p0.real, p0.imag]), bounds=bounds, method="Nelder-Mead")
    print(res)

    x = res.x.copy()
    print("Found: ", x)
    print("Goal: ", [p_goal.real, p_goal.imag])

    plt.show()
