import numpy as np
from numpy.fft import fft, fftfreq


def do_fft(t, y):
    dt = float(np.mean(np.diff(t)))
    freqs, data_fd = fftfreq(n=len(t), d=dt), fft(y)

    pos_slice = freqs >= 0

    return freqs[pos_slice], data_fd[pos_slice]
