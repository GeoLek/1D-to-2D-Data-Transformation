import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom

# Define global parameters for easy modification
EPS = 0.05  # Proximity threshold for recurrences
STEPS = 1   # Steps for considering recurrences

def load_eeg_data(file_path):
    """Load EEG data from a text file."""
    return np.loadtxt(file_path)

def create_recurrence_plot(data, eps=EPS, steps=STEPS):
    """Generate a recurrence plot from 1D EEG data."""
    n = len(data)
    recurrence_matrix = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i - steps, i + steps + 1):
            if 0 <= j < n:
                if abs(data[i] - data[j]) < eps:
                    recurrence_matrix[i, j] = 1
    return recurrence_matrix

def create_and_save_rp_images(set_label, record_name, input_file, output_folder, target_shape=(224, 224)):
    """Process EEG data files to create and save Recurrence Plot-based images."""
    os.makedirs(output_folder, exist_ok=True)

    # Load the EEG data
    eeg_data = load_eeg_data(input_file)

    # Generate recurrence plot matrix
    rp_matrix = create_recurrence_plot(eeg_data)
    resized_rp = zoom(rp_matrix, (target_shape[0] / rp_matrix.shape[0], target_shape[1] / rp_matrix.shape[1]), order=0)
    rp_image = Image.fromarray(resized_rp * 255)  # Convert binary matrix to an image

    # Save the plot as a PNG file with the annotation in the file name
    image_filename = f'{record_name}.png'
    rp_image.save(os.path.join(output_folder, image_filename))

    print(f'Saved image for {record_name} in {output_folder}')

# Define the base directory containing the EEG data
input_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Preprocessed data/Preprocessing 1'
output_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/EEG_Dimension Transformation Images/Recurrence Plots'

# Create output directories for each set and letter
sets = ['Set A', 'Set B', 'Set C', 'Set D', 'Set E']
letters = ['Z', 'O', 'N', 'F', 'S']

for set_name, letter in zip(sets, letters):
    input_dir = os.path.join(input_base_dir, set_name, letter)
    output_dir = os.path.join(output_base_dir, set_name, letter)
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each text file in the set directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_dir, filename)
            record_name = os.path.splitext(filename)[0]  # e.g., 'processed_Z001'
            create_and_save_rp_images(set_name, record_name, input_file, output_dir)

print("CWT transformation and image saving complete.")