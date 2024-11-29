# Script 4: Filtering, Baseline Removal, Normalization, Artifact rejection, smoothing and clipping

import os
import numpy as np
from scipy.signal import butter, filtfilt, detrend, medfilt
from sklearn.preprocessing import StandardScaler

# Define the input and output directories
input_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Set A/Z'
output_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Preprocessed data/Preprocessing 4/Set A/Z'
os.makedirs(output_dir, exist_ok=True)

# Filtering parameters
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=173.61, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Additional Processing Functions
def artifact_rejection(data, threshold=5):
    # Reject segments exceeding the threshold as artifacts
    return np.where(np.abs(data) > threshold, np.median(data), data)

def smooth_signal(data, kernel_size=5):
    # Apply median filter for smoothing
    return medfilt(data, kernel_size)

def clip_outliers(data, min_value=-3, max_value=3):
    # Clip values to reduce outliers
    return np.clip(data, min_value, max_value)

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

        # Step 4: Artifact Rejection
        artifact_free_data = artifact_rejection(normalized_data)

        # Step 5: Smoothing
        smoothed_data = smooth_signal(artifact_free_data)

        # Step 6: Clipping Outliers
        final_data = clip_outliers(smoothed_data)

        # Save the processed data
        output_file = os.path.join(output_dir, f'processed_{filename}')
        np.savetxt(output_file, final_data)

        print(f"Processed and saved: {output_file}")
