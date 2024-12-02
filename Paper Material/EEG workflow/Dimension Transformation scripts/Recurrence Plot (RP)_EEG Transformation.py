import os
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

def load_eeg_data(file_path):
    """Load EEG data from a text file."""
    return np.loadtxt(file_path)

def select_threshold(data, recurrence_rate=0.05):
    """Select the threshold epsilon based on a fixed recurrence rate."""
    data = data.reshape(-1, 1)
    # Compute all pairwise absolute differences (distances)
    distances = np.abs(data - data.T)
    # Extract the upper triangle of the distance matrix, excluding the diagonal
    distances = distances[np.triu_indices_from(distances, k=1)]
    # Determine the epsilon that achieves the desired recurrence rate
    epsilon = np.percentile(distances, recurrence_rate * 100)
    return epsilon

def create_recurrence_plot(data, epsilon):
    """Generate a recurrence plot from 1D EEG data by comparing all pairs of points."""
    data = data.reshape(-1, 1)
    # Compute the recurrence matrix based on the epsilon threshold
    distances = np.abs(data - data.T)
    recurrence_matrix = (distances < epsilon).astype(np.uint8)
    return recurrence_matrix

def create_and_save_rp_images(set_label, record_name, input_file, output_folder, target_shape=(224, 224)):
    """Process EEG data files to create and save recurrence plot images."""
    os.makedirs(output_folder, exist_ok=True)

    # Load the EEG data
    eeg_data = load_eeg_data(input_file)

    # Normalize the data to have zero mean and unit variance
    eeg_data = (eeg_data - np.mean(eeg_data)) / np.std(eeg_data)

    # Select threshold epsilon based on a fixed recurrence rate
    epsilon = select_threshold(eeg_data, recurrence_rate=0.05)

    # Generate the recurrence plot matrix
    rp_matrix = create_recurrence_plot(eeg_data, epsilon)

    # Resize the recurrence matrix to the target image size
    resized_rp = zoom(rp_matrix, (target_shape[0] / rp_matrix.shape[0],
                                  target_shape[1] / rp_matrix.shape[1]), order=0)

    # Convert the resized recurrence matrix to an image
    rp_image = Image.fromarray(resized_rp * 255).convert('L')  # Ensure it's in grayscale

    # Save the image as a PNG file
    image_filename = f'{record_name}.png'
    rp_image.save(os.path.join(output_folder, image_filename))

    print(f'Saved image for {record_name} in {output_folder}')

# Define the base directory containing the EEG data
input_base_dir = '/home/orion/Geo/Projects/1D-to-2D-Data-Transformation/Preprocessed data/Preprocessing 1'
output_base_dir = '/home/orion/Geo/Projects/1D-to-2D-Data-Transformation//Output_updated/Recurrence Plots'

# Create output directories for each set and corresponding letter
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

print("Recurrence plot transformation and image saving complete.")
