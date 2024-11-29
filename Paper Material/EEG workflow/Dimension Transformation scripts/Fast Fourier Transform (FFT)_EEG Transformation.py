import os
import numpy as np
from scipy.ndimage import zoom
from PIL import Image
from numpy.fft import fft, fftshift

# Parameters
TARGET_SHAPE = (224, 224)  # Target shape for images


def load_eeg_data(file_path):
    """Load EEG data from a .txt file."""
    return np.loadtxt(file_path)


def scale_to_grayscale(image_data):
    """Normalize the image data and convert it to grayscale."""
    normalized_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    grayscale_data = (normalized_data * 255).astype(np.uint8)
    return grayscale_data


def create_fft_image(eeg_data, target_shape=TARGET_SHAPE):
    """Create a FFT-based grayscale image from EEG data."""
    fft_data = fft(eeg_data)
    fft_data_shifted = fftshift(fft_data)  # Center the zero frequency component
    magnitude_spectrum = np.abs(fft_data_shifted)
    # Resize the 1D magnitude spectrum to fit the target shape
    height_scale = target_shape[0] / magnitude_spectrum.shape[0]
    scaled_spectrum = zoom(magnitude_spectrum, height_scale, order=1)
    # Convert to a 2D image by tiling
    scaled_spectrum_2d = np.tile(scaled_spectrum, (target_shape[1], 1)).T
    return scale_to_grayscale(scaled_spectrum_2d)


def process_and_save_fft_images(input_file, output_folder, target_shape=TARGET_SHAPE):
    """Process EEG files to create and save FFT-based images."""
    # Load EEG data from the text file
    eeg_data = load_eeg_data(input_file)

    # Create an FFT image
    fft_image = create_fft_image(eeg_data, target_shape)
    eeg_pil = Image.fromarray(fft_image)

    # Define output filename based on input file
    base_filename = os.path.basename(input_file).replace('.txt', '')
    image_filename = f'{base_filename}.png'

    # Save the FFT image
    eeg_pil.save(os.path.join(output_folder, image_filename))

    print(f'Saved FFT image for {base_filename} in {output_folder}')


# Define the input and output directories
input_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Preprocessed data/Preprocessing 1'
output_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/EEG_Dimension Transformation Images/Fast Fourier Transform (FFT)'

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
            process_and_save_fft_images(input_file_path, output_dir)

print("FFT transformation and image saving complete.")
