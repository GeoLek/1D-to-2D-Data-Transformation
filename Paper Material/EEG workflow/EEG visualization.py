import os
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the dataset and specific set folder (e.g., Set A)
dataset_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Preprocessed data/Preprocessing 4/Set A/Z'  # Change to your dataset path
example_file = 'processed_Z001.txt'  # Example file to visualize


# Load and plot the EEG data from a single file
def plot_eeg_signal(file_path, sampling_rate=173.61):
    # Load data from .txt file
    eeg_data = np.loadtxt(file_path)

    # Generate time axis in seconds
    time_axis = np.arange(0, len(eeg_data) / sampling_rate, 1 / sampling_rate)

    # Plot the signal
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, eeg_data, color='blue', linewidth=0.8)
    plt.title(f'EEG Signal from {os.path.basename(file_path)}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.grid(True)
    plt.show()


# Path to an example file for visualization
file_path = os.path.join(dataset_path, example_file)
plot_eeg_signal(file_path)
