# Script 3: Filtering, Baseline Removal, Normalization, Downsampling, and Epoch Segmentation

import os
import numpy as np
from scipy.signal import butter, filtfilt, detrend, decimate
from sklearn.preprocessing import StandardScaler

# Define the input and output directories
input_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Set A/Z'
output_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Preprocessed data/Preprocessing 3/Set A/Z'
os.makedirs(output_dir, exist_ok=True)

# Downsampling factor
downsample_factor = 2  # Adjust this factor as necessary

# Epoch duration (in seconds) and sampling rate after downsampling
epoch_duration = 5  # Set the epoch duration in seconds
sampling_rate = 173.61 / downsample_factor  # Adjusted sampling rate after downsampling
samples_per_epoch = int(epoch_duration * sampling_rate)

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

        # Step 4: Downsampling
        downsampled_data = decimate(normalized_data, downsample_factor)

        # Step 5: Epoch Segmentation
        num_epochs = len(downsampled_data) // samples_per_epoch
        for i in range(num_epochs):
            start_idx = i * samples_per_epoch
            end_idx = start_idx + samples_per_epoch
            epoch_data = downsampled_data[start_idx:end_idx]

            # Save each epoch as a separate file
            epoch_file = os.path.join(output_dir, f'processed_{filename}_epoch_{i}.txt')
            np.savetxt(epoch_file, epoch_data)
