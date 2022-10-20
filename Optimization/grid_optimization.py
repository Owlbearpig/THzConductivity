from imports import *
from Optimization.cost_function import Cost
from functools import partial


def main():
    d_list = [inf, 500, inf]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]

    new_cost = Cost(d_list, keywords, sam_idx=9, simulated_sample=False)
    freqs = new_cost.freqs

    vals = []
    for freq in freqs:
        if freq > 3.5:
            continue

        selected_freq_idx = np.argmin(np.abs(freqs - freq))
        print(f"Selected frequency: {freqs[selected_freq_idx]} (THz), idx: {selected_freq_idx}")

        cost_func = partial(new_cost.cost, selected_freq_idx)

        bounds = [[3.4, 3.9], [0.002, 0.020]]

        rez_x, rez_y = 300, 300
        grd_x = np.linspace(bounds[0][0], bounds[0][1], rez_x)
        grd_y = np.linspace(bounds[1][0], bounds[1][1], rez_y)

        grid_vals = np.zeros((rez_x, rez_y))

        for i in range(rez_x):
            for j in range(rez_y):
                p = array([grd_x[i], grd_y[j]])
                grid_vals[i, j] = cost_func(p)

        g_min_idx = np.argmin(grid_vals)
        min_x, min_y = np.unravel_index(g_min_idx, grid_vals.shape)

        p_found = [grd_x[min_x], grd_y[min_y]]
        print(f"Found: {p_found}, fx={cost_func(p_found)}")
        vals.append(p_found)
    np.save("grd_opt.npy", array(vals))

if __name__ == '__main__':
    main()
