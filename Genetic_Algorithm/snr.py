from main import path_dataset
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np

'''
Calculates the signal to noise ratio of the dataset
'''

file_path = path_dataset() 
df = pd.read_csv(file_path)
df = df.drop('Date', axis = 1)
dataset = df['Close'].values

signal = savgol_filter(dataset, window_length=11, polyorder=2)
noise = dataset - signal

signal_variance = np.var(signal)
noise_variance = np.var(noise)

snr = signal_variance / noise_variance
snr_db = 10 * np.log10(snr)

print(f"SNR: {snr}")
print(f"SNR (dB): {snr_db}")

'''
Output:
SNR: 432.8068251402451
SNR (dB): 26.362941007523375
'''