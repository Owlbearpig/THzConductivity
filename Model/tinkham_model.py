from Model.transmission_approximation import ri_approx
from scipy.constants import epsilon_0, c


def calc_sigma(ref_sub_fd, sam_sub_fd, ref_film_fd, sam_film_fd, d):
    T_sub = sam_sub_fd[:, 1] / ref_sub_fd[:, 1]
    T_film = sam_film_fd[:, 1] / ref_film_fd[:, 1]

    n_substr = ri_approx(ref_sub_fd, sam_sub_fd, d[0]*10**6)

    sigma = (epsilon_0 * c * (1 + n_substr[:, 1].real)) * (T_sub - T_film) / (T_film * d[1])

    return sigma

