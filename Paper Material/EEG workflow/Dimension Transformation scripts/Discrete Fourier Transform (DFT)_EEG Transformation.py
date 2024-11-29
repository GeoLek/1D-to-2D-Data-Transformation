import os
import numpy as np
from scipy.ndimage import zoom
from PIL import Image

# Define global parameters
TARGET_SHAPE = (224, 224)  # Target size for images


def load_eeg_data(file_path):
    """Load EEG data from a .txt file."""
    return np.loadtxt(file_path)


def apply_dft(eeg_data):
    """Apply Discrete Fourier Transform (DFT) to the EEG data."""
    dft_data = np.fft.fft(eeg_data)
    dft_shifted = np.fft.fftshift(dft_data)  # Shift the zero frequency component to the center
    magnitude_spectrum = np.abs(dft_shifted)
    return magnitude_spectrum


def scale_to_grayscale(image_data):
    """Normalize the image data and convert it to grayscale."""
    normalized_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    grayscale_data = (normalized_data * 255).astype(np.uint8)
    return grayscale_data


def create_dft_image(eeg_data, target_shape=TARGET_SHAPE):
    """Create a DFT-based grayscale image from EEG data."""
    dft_data = apply_dft(eeg_data)
    # Assuming DFT data is 1D, convert it to 2D by repeating it to match the target shape
    dft_image_2d = np.tile(dft_data, (target_shape[0], 1))
    # Scale the 2D image to the target size
    scaled_image = zoom(dft_image_2d, (1, target_shape[1] / dft_image_2d.shape[1]), order=1)
    return scale_to_grayscale(scaled_image)


def process_and_save_dft_images(input_file, output_folder, target_shape=TARGET_SHAPE):
    """Process EEG files to create and save DFT-based images."""
    # Load EEG data from the text file
    eeg_data = load_eeg_data(input_file)

    # Create a DFT image
    dft_image = create_dft_image(eeg_data, target_shape)
    eeg_pil = Image.fromarray(dft_image)

    # Define output filename based on input file
    base_filename = os.path.basename(input_file).replace('.txt', '')
    image_filename = f'{base_filename}.png'

    # Save the DFT image
    eeg_pil.save(os.path.join(output_folder, image_filename))

    print(f'Saved DFT image for {base_filename} in {output_folder}')


# Define the input and output directories
input_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Preprocessed data/Preprocessing 1'
output_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/EEG_Dimension Transformation Images/Discrete Fourier Transform (DFT)'

# Create output directories for each set and letter
sets = ['Set A', 'Set B', 'Set C', 'Set D', 'Set E']
letters = ['Z', 'O', 'N', 'F', 'S']

for set_name, letter in zip(sets, letters):
    input_dir = os.path.join(input_base_dir, set_name, letter)
    output_dir = os.path.join(output_base_dir, set_name, letter)
    os.makedirs(output_dir, exist_ok=True)

    # Process each txt file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_dir, filename)
            process_and_save_dft_images(input_file_path, output_dir)

print("DFT transformation and image saving complete.")
