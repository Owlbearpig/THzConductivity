import numpy as np
from datetime import datetime
from consts import data_dir
from numpy.fft import fft, fftfreq


class Measurement:
    def __init__(self, filepath, post_process_config):
        self.filepath = filepath
        self.dir_1above, self.dir_2above = self.filepath.parents[0], self.filepath.parents[1]
        self.meas_time = None
        self.meas_type = None
        self.sample_name = None
        self.position = [None, None]
        self._data = None

        self.post_process_config = post_process_config

        self.set_metadata()

    def __repr__(self):
        return str(self.filepath)

    def set_metadata(self):
        # set time
        date_string = str(self.filepath.stem)[:25]
        self.meas_time = datetime.strptime(date_string, "%Y-%m-%dT%H-%M-%S.%f")

        # set sample name
        if ("sam" in self.dir_1above.stem.lower()) or ("ref" in self.dir_1above.stem.lower()):
            self.sample_name = self.dir_2above.stem
        else:
            self.sample_name = self.dir_1above.stem

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
        if self._data is None:
            self._data = np.loadtxt(self.filepath)
        if self.post_process_config["sub_offset"]:
            self._data[:, 1] -= np.mean(self._data[:10, 1])

        return self._data

    def get_data_fd(self):
        data_td = self.get_data_td()
        t, y = data_td[:, 0], data_td[:, 1]
        dt = float(np.mean(np.diff(t)))
        freqs, data_fd = fftfreq(n=len(t), d=dt), fft(y)

        return freqs, data_fd


def get_measurements():
    post_process_config = {"sub_offset": True, }

    measurements = []

    glob = data_dir.glob("**/*")
    for file_path in glob:
        if file_path.is_file():
            try:
                measurements.append(Measurement(file_path, post_process_config))
            except ValueError:
                print(f"Skipping, not a measurement: {file_path}")

    return measurements



