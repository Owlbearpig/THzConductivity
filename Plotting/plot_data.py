from imports import *
from functions import phase_correction, do_ifft


def plot(data_fd, label=""):
    if "mod" in label.lower():
        color = "black"
    else:
        color = None

    freqs = data_fd[:, 0]

    plt.figure("Wrapped phase")
    plt.title("Wrapped phase")
    plt.plot(freqs, np.angle(data_fd[:, 1]), label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (rad)")
    plt.legend()

    phase_unwrapped = phase_correction(data_fd)
    plt.figure("Unwrapped phase")
    plt.title("Unwrapped phase")
    plt.plot(freqs, phase_unwrapped, label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (rad)")
    plt.legend()

    plt.figure("Spectrum")
    plt.title("Spectrum")
    plt.plot(freqs, np.abs(data_fd[:, 1]), label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()

    data_td = do_ifft(data_fd, hermitian=True)

    plt.figure("Time domain")
    plt.title("Time domain")
    plt.plot(data_td[:, 0], data_td[:, 1], label=label, color=color)
    plt.xlabel("Time (ps)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()

    if False:
        plt.figure("Real part")
        plt.title("Real part")
        plt.plot(data_fd[:, 0], data_fd[:, 1].real, label=label, color=color)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("real(E)")
        plt.legend()

        plt.figure("Imag part")
        plt.title("Imag part")
        plt.plot(data_fd[:, 0], data_fd[:, 1].imag, label=label, color=color)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Imag(E)")
        plt.legend()


if __name__ == '__main__':
    pass