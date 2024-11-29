# Script 1: Filtering, Baseline Removal, and Normalization

import os
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from sklearn.preprocessing import StandardScaler

# Define the input and output directories
input_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Set A/Z'
output_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Preprocessed data/Preprocessing 1/Set A/Z'
os.makedirs(output_dir, exist_ok=True)

# Filtering parameters
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=173.61, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Process each file
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_dir, filename)
        data = np.loadtxt(file_path)

        # Step 1: Apply Bandpass Filter
        filtered_data = bandpass_filter(data)

        # Step 2: Baseline Removal
        baseline_removed_data = detrend(filtered_data, type='constant')

        # Step 3: Normalization
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(baseline_removed_data.reshape(-1, 1)).flatten()

        # Save the processed data
        output_file = os.path.join(output_dir, f'processed_{filename}')
        np.savetxt(output_file, normalized_data)
