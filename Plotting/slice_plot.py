"""

3D image, with frequency on z axis -> slider

"""

from imports import *
from helpers import get_closest_idx
from Optimization.cost_function import Cost
from functools import partial
from matplotlib.widgets import Slider


def main():
    d_list = [inf, 500, inf]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]
    sample_idx = 9  # if None avg. will be used
    sim_sample = False
    file_name = "_".join(keywords) + f"_{sample_idx}" + ".npy"

    new_cost = Cost(d_list, keywords, sam_idx=sample_idx, simulated_sample=sim_sample)
    freqs = new_cost.freqs
    freq_range = [0.25, 1.5]
    freq_slice = (freq_range[0] <= freqs) * (freqs <= freq_range[1])

    bounds = [[3.4, 3.9], [0.002, 0.020]]

    rez_x, rez_y, rez_z = 300, 300, sum(freq_slice)
    grd_x = np.linspace(bounds[0][0], bounds[0][1], rez_x)
    grd_y = np.linspace(bounds[1][0], bounds[1][1], rez_y)
    grd_z = freqs[freq_slice]

    try:
        grid_vals = np.load(file_name)
    except FileNotFoundError:
        grid_vals = np.zeros((rez_x, rez_y, rez_z))
        for k, freq in enumerate(grd_z):
            print(f"Selected frequency: {freq} (THz), idx: {k} / {rez_z}")
            cost_func = partial(new_cost.cost, freq)

            for i in range(rez_x):
                for j in range(rez_y):
                    p = array([grd_x[i], grd_y[j]])
                    grid_vals[i, j, k] = cost_func(p)

        np.save(file_name, grid_vals)

    grid_vals = np.log10(grid_vals)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Residual sum plot")
    fig.subplots_adjust(left=0.2)
    extent = [grd_x[0], grd_x[-1], grd_y[0], grd_y[-1]]
    aspect = ((bounds[0][1] - bounds[0][0]) / rez_x) / ((bounds[1][1] - bounds[1][0]) / rez_y)
    img = ax.imshow(grid_vals[:, :, 0].transpose((1, 0)),
                    vmin=np.min(grid_vals), vmax=np.max(grid_vals),
                    origin="lower",
                    cmap=plt.get_cmap("jet"),
                    extent=extent,
                    aspect=aspect)

    ax.set_xlabel("n")
    ax.set_ylabel("k")

    """
    n, k = [], []
    for freq_idx, freq in enumerate(grd_z):
        g_min_idx = np.argmin(grid_vals[:, :, freq_idx])
        min_x, min_y = np.unravel_index(g_min_idx, grid_vals.shape)
        cost_func = partial(new_cost.cost, freq)
        p_found = [grd_x[min_x], grd_y[min_y]]
        print(f"Found: {p_found}, fx={cost_func(p_found)}")
        n.append(grd_x[min_x]), k.append(grd_y[min_y])
    """

    cbar = fig.colorbar(img)
    cbar.set_label("log10(loss)", rotation=270, labelpad=20)
    axmax = fig.add_axes([0.05, 0.1, 0.02, 0.8])
    amp_slider = Slider(
        ax=axmax,
        label="Frequency (THz)",
        valstep=grd_z * um,
        valmin=grd_z[0] * um,
        valmax=grd_z[-1] * um,
        valinit=grd_z[0] * um,
        orientation="vertical"
    )

    def update(val):
        idx, = np.where(grd_z == val)
        img.set_data(grid_vals[:, :, idx].transpose((1, 0, 2)))
        fig.canvas.draw()

    amp_slider.on_changed(update)


if __name__ == '__main__':
    main()
    plt.show()

