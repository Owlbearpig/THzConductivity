import numpy as np
from datetime import datetime
from consts import data_dir
from numpy.fft import fft, fftfreq
from functions import windowing


class Measurement:
    def __init__(self, data=None, meas_type=None, filepath=None, post_process_config=None):
        if data is not None:
            self._data_td, self._data_fd = data, None
        else:
            self._data_td, self._data_fd = None, None

        self.filepath = filepath

        self.meas_time = None
        self.meas_type = None
        self.sample_name = None
        self.position = [None, None]

        if post_process_config is None:
            from consts import post_process_config

        self.post_process_config = post_process_config

        self.set_metadata(meas_type)

    def __repr__(self):
        return str(self.filepath)

    def set_metadata(self, meas_type=None):
        if meas_type is not None:
            self.meas_type = meas_type
            return

        # set time
        date_string = str(self.filepath.stem)[:25]
        self.meas_time = datetime.strptime(date_string, "%Y-%m-%dT%H-%M-%S.%f")

        # set sample name
        dir_1above, dir_2above = self.filepath.parents[0], self.filepath.parents[1]
        if ("sam" in dir_1above.stem.lower()) or ("ref" in dir_1above.stem.lower()):
            self.sample_name = dir_2above.stem
        else:
            self.sample_name = dir_1above.stem

        # set measurement type
        if "ref" in str(self.filepath.stem).lower():
            self.meas_type = "ref"
        elif "sam" in str(self.filepath.stem).lower():
            self.meas_type = "sam"
        else:
            self.meas_type = "other"

        # set position
        str_splits = str(self.filepath).split("_")
        x = float(str_splits[-2].split(" mm")[0])
        y = float(str_splits[-1].split(" mm")[0])
        self.position = [x, y]

    def get_data_td(self):
        if self._data_td is None:
            self._data_td = np.loadtxt(self.filepath)
        if self.post_process_config["sub_offset"]:
            self._data_td[:, 1] -= np.mean(self._data_td[:10, 1])
        if self.post_process_config["en_windowing"]:
            self._data_td[:, 1] = windowing(self._data_td[:, 1])

        return self._data_td

    def get_data_fd(self, pos_freqs_only=True):
        if self._data_fd is not None:
            return self._data_fd
        data_td = self.get_data_td()
        t, y = data_td[:, 0], data_td[:, 1]
        dt = float(np.mean(np.diff(t)))
        freqs, data_fd = fftfreq(n=len(t), d=dt), fft(y)

        if pos_freqs_only:
            pos_slice = freqs >= 0
            self._data_fd = np.array([freqs[pos_slice], data_fd[pos_slice]]).T
        else:
            self._data_fd = np.array([freqs, data_fd]).T

        return self._data_fd


def get_all_measurements():
    measurements = []

    glob = data_dir.glob("**/*")
    for file_path in glob:
        if file_path.is_file():
            try:
                measurements.append(Measurement(filepath=file_path))
            except ValueError:
                print(f"Skipping, not a measurement: {file_path}")

    return measurements


def avg_data(measurements):
    data_0 = measurements[0].get_data_td()
    t = data_0[:, 0]

    y_arrays = []
    for measurement in measurements:
        data_td = measurement.get_data_td()
        y_arrays.append(data_td[:, 1])

    return np.array([t, np.mean(y_arrays, axis=0)]).T

