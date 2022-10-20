
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
    plt.plot_field(td_sam_mod[:, 0], td_sam_mod[:, 1], label="Sam. model", color="black")
    plt.plot_field(self.ref_data_td[:, 0], self.ref_data_td[:, 1], label="Ref. measurement")
    plt.plot_field(self.sam_data_td[:, 0], self.sam_data_td[:, 1], label="Sam. measurement")
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


def cost_cauchy_relation(self, p, en_plot=False):
    # A = 3.658, B = 0.000354 (very approximate values)
    n = cauchy_relation(self.freqs, p)

    t = tmm_package_wrapper(self.freqs, self.d_list, n)

    mod_fd = array([self.ref_data_fd[:, 0], t * self.ref_data_fd[:, 1]]).T

    if en_plot:
        plt.figure("Refractive index, cauchy relation")
        plt.title("Refractive index, cauchy relation")
        plt.plot_field(self.freqs, n)

        plot(mod_fd, label="model")
        plot(self.sam_data_fd, label="sam")

    mod_phase_unwrapped = phase_correction(mod_fd)

    eval_range = get_closest_idx(self.freqs, 0.20), get_closest_idx(self.freqs, 3.00)

    loss = np.sum((mod_phase_unwrapped[eval_range[0]:eval_range[1]] -
                   self.sam_phase_unwrapped[eval_range[0]:eval_range[1]]) ** 2)

    return loss

""" Custom step-function """
class RandomDisplacementBounds(object):
    """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
        Modified! (dropped acceptance-rejection sampling for a more specialized approach)
    """
    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds """
        min_step = np.maximum(self.xmin - x, -self.stepsize)
        max_step = np.minimum(self.xmax - x, self.stepsize)

        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step

        return xnew