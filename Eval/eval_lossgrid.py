"""

3D image, with frequency on z axis -> slider

"""
import matplotlib.pyplot as plt
import numpy as np

from imports import *
from Measurements.measurements import select_measurements
from Optimization.cost_function import Cost
from Plotting.plot_data import plot_ri, plot_field
from Model.tmm_package import tmm_from_ri


def m_smallest_positions(arr, m):
    idx = np.argpartition(arr.flatten(), m)
    idx_unraveled = np.unravel_index(idx, arr.shape)
    # arr.ndim == number of axes in idx_unraveled ->
    # e.g. 2D; p1 = (idx_unraveled[0,0], idx_unraveled[1,0])

    return idx_unraveled


def main():
    d_list = [inf, 500, inf]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]
    sample_idx = 3
    new_cost = Cost(d_list, keywords)
    freqs = new_cost.freqs
    freq_range = [0.25, 3.5]
    freq_slice = (freq_range[0] <= freqs) * (freqs <= freq_range[1])

    bounds = [[3.4, 3.9], [0.002, 0.020]]

    rez_x, rez_y, rez_z = 300, 300, sum(freq_slice)
    grd_x = np.linspace(bounds[0][0], bounds[0][1], rez_x)
    grd_y = np.linspace(bounds[1][0], bounds[1][1], rez_y)
    grd_z = freqs[freq_slice]

    n_vals, k_vals = [], []
    grid_vals = np.load(str(ROOT_DIR / "Plotting" / f"01 GaAs Wafer 25_2022_02_14_{sample_idx}_False.npy"))
    for k, freq in enumerate(grd_z):
        img = grid_vals[:, :, k]
        n_idx, k_idx = m_smallest_positions(img, 1)
        n, k = grd_x[n_idx[:1]], grd_y[k_idx[:1]]
        n_vals.append(sorted(n))
        k_vals.append(sorted(k))

    n_vals = array(n_vals).flatten()
    k_vals = array(k_vals).flatten()

    n = array([grd_z, n_vals + 1j * k_vals]).T

    #plot_ri(n, label=f"Sample {sample_idx} grid minima")

    refs, sams = select_measurements(keywords)
    ref, sam = refs[sample_idx], sams[sample_idx]
    ref_fd, sam_fd = ref.get_data_fd(), sam.get_data_fd()

    plot_field(ref_fd, label=f"ref_{sample_idx}")
    plot_field(sam_fd, label=f"sample_{sample_idx}")

    mod_fd = tmm_from_ri(n, d_list, ref_fd, en_plot=True)

    plot_field(mod_fd, label=f"TMM(n_grid_minima)")


if __name__ == '__main__':
    main()
    plt.show()