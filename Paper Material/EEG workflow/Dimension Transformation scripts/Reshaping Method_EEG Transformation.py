import os
import numpy as np
from scipy.ndimage import zoom
from PIL import Image


def load_eeg_data(file_path):
    """Load EEG data from a text file."""
    return np.loadtxt(file_path)


def scale_to_grayscale(data):
    """Scale data to grayscale (0 to 255)."""
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    grayscale_data = (normalized_data * 255).astype(np.uint8)
    return grayscale_data


def create_reshaped_image(data, target_shape=(224, 224)):
    """Convert 1D data into a 2D reshaped grayscale image."""
    # Convert to grayscale and reshape to fit the target shape
    grayscale_data = scale_to_grayscale(data)
    height_scale = target_shape[0] / len(grayscale_data)

    # Stretch the signal vertically
    stretched_signal = zoom(grayscale_data, height_scale, order=1)

    # Determine the width required to match target shape
    width_scale = target_shape[1] / len(stretched_signal)
    stretched_signal_width = zoom(stretched_signal, width_scale, order=1)

    # Create the 2D image with the signal centered
    image_2d = np.zeros(target_shape, dtype=np.uint8)
    start_row = (target_shape[0] - stretched_signal.shape[0]) // 2
    start_col = (target_shape[1] - stretched_signal_width.shape[0]) // 2
    image_2d[start_row:start_row + stretched_signal.shape[0],
    start_col:start_col + stretched_signal_width.shape[0]] = stretched_signal_width

    return image_2d


def create_and_save_images(set_label, record_name, input_file, output_folder, target_shape=(224, 224)):
    """Process EEG files to create and save reshaped images."""
    os.makedirs(output_folder, exist_ok=True)

    # Load EEG data
    eeg_data = load_eeg_data(input_file)
    reshaped_image = create_reshaped_image(eeg_data, target_shape)

    # Convert to PIL Image and resize to target dimensions
    image = Image.fromarray(reshaped_image)
    image = image.resize(target_shape, Image.BILINEAR)

    # Save the image
    image_filename = f"{record_name}.png"
    image.save(os.path.join(output_folder, image_filename))
    print(f"Saved image for {record_name} in {output_folder}")


# Define base directories for input and output data
input_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Preprocessed data/Preprocessing 1'
output_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/EEG_Dimension Transformation Images/Reshaping Method'

# Ensure output directories for each set and letter exist
sets = ['Set A', 'Set B', 'Set C', 'Set D', 'Set E']
letters = ['Z', 'O', 'N', 'F', 'S']

for set_name, letter in zip(sets, letters):
    input_dir = os.path.join(input_base_dir, set_name, letter)
    output_dir = os.path.join(output_base_dir, set_name, letter)
    os.makedirs(output_dir, exist_ok=True)

    # Process each EEG text file in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_dir, filename)
            record_name = os.path.splitext(filename)[0]
            create_and_save_images(set_name, record_name, input_file, output_dir)

print("Reshaping and image saving complete.")
