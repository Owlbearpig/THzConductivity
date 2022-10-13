from Measurements.measurements import get_all_measurements
import matplotlib.pyplot as plt
import numpy as np
from functions import do_fft

measurements = get_all_measurements()

keywords = ["GaAs", "Wafer", "01"]
#keywords = ["InP Wafer 5"]

selected_measurements = []
for measurement in measurements:
    #print(measurement)
    if all([s in str(measurement.filepath) for s in keywords]):
        selected_measurements.append(measurement)
        print(measurement)


samples = [x for x in selected_measurements if x.meas_type == "sam"]
refs = [x for x in selected_measurements if x.meas_type == "ref"]

sam_avg = np.mean([x.get_data_td()[:, 1] for x in samples], axis=0)
ref_avg = np.mean([x.get_data_td()[:, 1] for x in refs], axis=0)

t = samples[0].get_data_td()[:, 0]
y = samples[0].get_data_td()[:, 1]
y_ref = refs[0].get_data_td()[:, 1]

y[900:] = y[900]*0
sam_avg[900:] = sam_avg[900:]*0
ref_avg[900:] = ref_avg[900:]*0


freqs, Y_single = do_fft(t, y)
_, Y_single_ref = do_fft(t, y_ref)
_, Y_avg_sam = do_fft(t, sam_avg)
_, Y_avg_ref = do_fft(t, ref_avg)

t_func = Y_avg_sam / Y_avg_ref

plt.figure()
plt.plot(t, y, label="sam single")
plt.plot(t, sam_avg, label="sam average")
plt.plot(t, ref_avg, label="ref average")
plt.legend()

plt.figure()

plt.plot(freqs, (np.abs(Y_single)), label="sam single")
plt.plot(freqs, (np.abs(Y_single_ref)), label="sam single ref")
plt.plot(freqs, (np.abs(Y_avg_sam)), label="sam average")
plt.plot(freqs, (np.abs(Y_avg_ref)), label="ref average")
plt.legend()

plt.figure()
plt.plot(freqs, (t_func.real), label="transfer func")
plt.legend()

plt.show()
