from imports import *
from functions import calc_absorption, do_ifft
from cost_function import Cost
from functools import partial
from helpers import get_closest_idx, most_frequent
from scipy.optimize import shgo, basinhopping, minimize
from Model.transmission_approximation import ri_approx
from Model.tmm_package import tmm_package_wrapper


def main():
    d_list = [inf, 500, inf]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]

    new_cost = Cost(d_list, keywords, simulated_sample=False)
    freqs = new_cost.freqs

    freq_range = [0.25, 1.6]
    freq_slice = (freq_range[0] < freqs)*(freqs < freq_range[1])
    freqs = freqs[freq_slice]

    bounds = [(3.3, 3.8), (0.001, 0.02)]
    #print(get_closest_idx(freqs, 0.614846288427893))
    n = []
    for idx, freq in enumerate(freqs):
        if idx != 72:
            pass
        print(f"Optimizing frequency {freq} / {freqs.max()} THz ({idx}/{len(freqs)})")

        cost_func = partial(new_cost.cost, float(freq))

        minimizer_kwargs = {"tol": 1e-14, "method": "Nelder-Mead", "bounds": bounds}
        res = shgo(cost_func, bounds=bounds, n=500, iters=5, minimizer_kwargs=minimizer_kwargs)

        #res = minimize(cost_func, p0, bounds=bounds)
        """
        minimizer_kwargs = {"bounds": bounds}
        res = basinhopping(cost_func, p0, 100, 1, stepsize=0.005, minimizer_kwargs=minimizer_kwargs, disp=False)
        """

        n.append(res.x[0] + 1j * res.x[1])
        print(f"Result: {n[-1]}")

    n = array(n)

    a_fit = calc_absorption(freqs, n.imag)
    np.save("freqs_" + "_".join(keywords), freqs)
    np.save("n_" + "_".join(keywords), n)

    goal_freq_slice = (freq_range[0] < new_cost.freqs) * (new_cost.freqs < freq_range[1])
    freq_goal = new_cost.freqs[goal_freq_slice]
    n_goal = new_cost.n_approx[goal_freq_slice]

    plt.figure()
    plt.title("Refractive index")
    plt.plot(freqs, n.real, label="RI fit")
    plt.plot(freq_goal, n_goal.real, label="RI approximation")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Refractive index")
    plt.legend()

    plt.figure()
    plt.title("Extinction coefficient")
    plt.plot(freqs, n.imag, label="Extinction coeff. fit")
    plt.plot(freq_goal, n_goal.imag, label="Extinction coeff. approximation")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Extinction coefficient")

    plt.figure()
    plt.title("Absorption coefficient")
    plt.plot(freqs, a_fit, label="Absorption coeff. fit")
    plt.plot(freq_goal, calc_absorption(freq_goal, n_goal.imag),
             label="Absorption coeff. approximation")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Absorption coefficient (1/cm)")

    t = tmm_package_wrapper(freqs, d_list, n)

    mod_fd = array([freqs, new_cost.ref_data_fd[goal_freq_slice, 1] * t]).T
    mod_td = do_ifft(mod_fd, hermitian=True)

    plt.figure()
    plt.title("Time domain")
    plt.plot(mod_td[:, 0], mod_td[:, 1], label="Sam. model", color="black")
    plt.plot(new_cost.ref_data_td[:, 0], new_cost.ref_data_td[:, 1], label="Ref. measurement")
    plt.plot(new_cost.sam_data_td[:, 0], new_cost.sam_data_td[:, 1], label="Sam. measurement")
    plt.xlabel("Time (ps)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()


if __name__ == '__main__':
    main()
    plt.show()
