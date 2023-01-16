""""
Discard samples (one minute intervals) with to much noise.
Input: labeled ECG data, with every row a label in the first column followed by the one minute of ECG data in the remaining columns.
ouput: noise free labeled ECG, data (same format, just less rows assuming there is at least 1 noisy minute)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.fft import fft, fftfreq

# Load data and transform to dataframe
path = "./physionet_apnea/all_subjects.csv"
df = pd.read_csv(path)
data_arr = np.array(df)

# Add noise free labeled ECG data to list D based on the ratio between the sum of the fft between 0.7 and 3 Hz and
# the sum of the fft over the whole frequency interval ([0,fs/2] = [0,50]Hz)
fs = 100
D = []
for irow in range(data_arr.shape[0]):
    ecg_minute = data_arr[irow, 1:]
    max_amplitude = np.max(ecg_minute)

    N = len(ecg_minute)
    T = N / fs
    t = np.linspace(0, T, N)

    ecg_fft = fft(ecg_minute)
    xf = np.linspace(0, fs, N)

    amp_d7_3_Hz = sum(np.abs(ecg_fft[N // 143:N // 33]))
    amp_sum = sum(np.abs(ecg_fft[0:N // 2]))
    ratio = amp_d7_3_Hz / amp_sum

    if (ratio > 0.084):  # add "good" data to D
        D.append(data_arr[irow, :])

D_arr = np.array(D)

# Save the noise free data to csv
df = pd.DataFrame(D_arr)
df.to_csv(path+"/all_subjects_noise_free", header=False, index=False)



