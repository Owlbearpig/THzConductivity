"""
run local optimization at each grid point p0 for one frequency
until fx doesn't change between iterations. Map this fx on the grid.
"""
from imports import *
from functions import calc_absorption
from Optimization.cost_function import Cost
from functools import partial
from Model.transmission_approximation import ri_approx
from scipy.optimize import minimize


def main():
    d_list = [inf, 500, inf]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]

    simulated_sample = True
    new_cost = Cost(d_list, keywords, simulated_sample=simulated_sample)
    freqs = new_cost.freqs
    freq = 1.300
    selected_freq_idx = np.argmin(np.abs(freqs - freq))
    print(f"Selected frequency: {freqs[selected_freq_idx]} (THz), idx: {selected_freq_idx}")

    file_name = "_".join(keywords) + f"_p0_grid_opt_{freq}_{simulated_sample}" + ".npy"

    cost_func = partial(new_cost.cost, selected_freq_idx)

    bounds = [[3.62, 3.68], [0.001, 0.020]]

    rez_x, rez_y = 300, 300
    grd_x = np.linspace(bounds[0][0], bounds[0][1], rez_x)
    grd_y = np.linspace(bounds[1][0], bounds[1][1], rez_y)

    grid_vals = np.zeros((rez_x, rez_y))

    minimizer_kwargs = {"maxiter": np.inf, "maxfev": np.inf, "xatol": 1e-15, "fatol": np.inf,
                        "return_all": False}

    for i in range(rez_x):
        if i % 10 == 0:
            print(f"{i} / {rez_x}")
        for j in range(rez_y):
            p0 = array([grd_x[i], grd_y[j]])
            res = minimize(cost_func, p0, method="Nelder-Mead", bounds=bounds, options=minimizer_kwargs)
            grid_vals[i, j] = res.fun
            # print(res)

    np.save(file_name, grid_vals)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Residual sum plot")
    fig.subplots_adjust(left=0.2)
    extent = [grd_x[0], grd_x[-1], grd_y[0], grd_y[-1]]
    aspect = ((bounds[0][1] - bounds[0][0]) / rez_x) / ((bounds[1][1] - bounds[1][0]) / rez_y)
    img = ax.imshow(grid_vals[:, :].transpose((1, 0)), vmin=np.min(grid_vals), vmax=np.max(grid_vals),
                    origin="lower",
                    cmap=plt.get_cmap("jet"),
                    extent=extent,
                    aspect=aspect)

    ax.set_xlabel("n")
    ax.set_ylabel("k")

    cbar = fig.colorbar(img)
    cbar.set_label("min. fun. val.", rotation=270, labelpad=20)


if __name__ == '__main__':
    main()
    plt.show()
