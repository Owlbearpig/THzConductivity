from Measurements.measurements import get_measurements
import matplotlib.pyplot as plt
import numpy as np


measurements = get_measurements()

sample = "p-doped GaAs_C 18817"

for measurement in measurements:
    if sample in measurement.sample_name:
        print(measurement)
