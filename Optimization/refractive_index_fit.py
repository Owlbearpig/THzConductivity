from imports import *
from functions import calc_absorption, do_ifft
from cost_function import Cost
from functools import partial
from scipy.optimize import shgo, basinhopping, minimize
from Model.tmm_package import tmm_package_wrapper
from Plotting.plot_data import plot_ri, plot_field


def main():
    d_list = [inf, 500, inf]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]

    new_cost = Cost(d_list, keywords, simulated_sample=True, en_noise=True)
    freqs = new_cost.freqs

    freq_range = [0.00, 10.00]
    freq_slice = (freq_range[0] <= freqs) * (freqs <= freq_range[1])
    freqs = freqs[freq_slice]

    bounds = [(3.62, 3.68), (0.001, 0.02)]
    # print(get_closest_idx(freqs, 0.614846288427893))
    n, freq_evaluated = [], []
    for idx, freq in enumerate(freqs):
        if idx > 100:
            pass
        print(f"Optimizing frequency {freq} / {freqs.max()} THz ({idx}/{len(freqs)})")
        #freq_evaluated.append(freq)

        cost_func = partial(new_cost.cost, float(freq))

        #minimizer_kwargs = {"tol": 1e-14, "bounds": bounds}
        #res = shgo(cost_func, bounds=bounds, n=500, iters=5, minimizer_kwargs=minimizer_kwargs)
        n_goal = new_cost.n_approx[idx, 1]
        p0 = array([n_goal.real, n_goal.imag])
        res = minimize(cost_func, p0, bounds=bounds, tol=1e-14)
        """
        minimizer_kwargs = {"bounds": bounds}
        res = basinhopping(cost_func, p0, 100, 1, stepsize=0.005, minimizer_kwargs=minimizer_kwargs, disp=False)
        """

        n.append(res.x[0] + 1j * res.x[1])
        print(f"Result: {n[-1]}")

    n = array([freqs, n]).T

    goal_freq_slice = (freq_range[0] <= new_cost.freqs) * (new_cost.freqs <= freq_range[1])
    n_goal = new_cost.n_approx[goal_freq_slice]

    plot_ri(n, label="RI fit")
    plot_ri(n_goal, label="RI approximation (truth if simulated sam.)")

    t = tmm_package_wrapper(d_list, n)

    mod_fd = array([freqs, new_cost.ref_data_fd[goal_freq_slice, 1] * t[:, 1]]).T


    plot_field(mod_fd, label="Sam. model", color="black")
    plot_field(new_cost.ref_data_fd, label="Ref. measurement")
    plot_field(new_cost.sam_no_noise, label="Sam. before adding noise")
    plot_field(new_cost.sam_data_fd, label="Sam. measurement")



if __name__ == '__main__':
    main()
    plt.show()
