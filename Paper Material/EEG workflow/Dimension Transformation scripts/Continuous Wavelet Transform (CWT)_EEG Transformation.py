import os
import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import zoom, gaussian_filter
from PIL import Image
from skimage import exposure

# Define global parameters for easy modification
WAVELET_NAME = 'morl'  # Options: 'morl', 'gaus8', 'gaus4', etc.
SIGMA = 0.1  # Gaussian smoothing parameter
CLIP_LIMIT = 0.03  # Contrast enhancement parameter


def load_eeg_data(file_path):
    """Load EEG data from a .txt file."""
    return np.loadtxt(file_path)


def apply_cwt(eeg_data, scales, wavelet_name=WAVELET_NAME):
    coefficients, frequencies = pywt.cwt(eeg_data, scales, wavelet_name)
    return np.abs(coefficients)


def scale_to_grayscale(image_data):
    normalized_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    grayscale_data = (normalized_data * 255).astype(np.uint8)
    return grayscale_data


def enhance_contrast(image_data, clip_limit=CLIP_LIMIT):
    img_eq = exposure.equalize_adapthist(image_data, clip_limit=clip_limit)
    return (img_eq * 255).astype(np.uint8)


def smooth_image(image_data, sigma=SIGMA):
    return gaussian_filter(image_data, sigma=sigma)


def create_cwt_image(eeg_data, target_shape=(224, 224)):
    scales = np.arange(1, 128)
    cwt_magnitude = apply_cwt(eeg_data, scales)
    resized_cwt = zoom(cwt_magnitude,
                       (target_shape[0] / cwt_magnitude.shape[0], target_shape[1] / cwt_magnitude.shape[1]), order=1)
    grayscale_image = scale_to_grayscale(resized_cwt)
    contrast_enhanced_image = enhance_contrast(grayscale_image)
    return smooth_image(contrast_enhanced_image)


def process_and_save_cwt_images(input_file, output_folder, target_shape=(224, 224)):
    """Process EEG file to create and save CWT-based image."""
    # Load EEG data from the text file
    eeg_data = load_eeg_data(input_file)

    # Create a CWT image
    cwt_image = create_cwt_image(eeg_data, target_shape)
    eeg_pil = Image.fromarray(cwt_image)

    # Define output filename based on input file
    base_filename = os.path.basename(input_file).replace('.txt', '')
    image_filename = f'{base_filename}.png'

    # Save the CWT image
    eeg_pil.save(os.path.join(output_folder, image_filename))

    print(f'Saved CWT image for {base_filename} in {output_folder}')


# Define the input and output directories
input_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Preprocessed data/Preprocessing 1'
output_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/EEG_Dimension Transformation Images/Continuous Wavelet Transform (CWT)'

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
            process_and_save_cwt_images(input_file_path, output_dir)

print("CWT transformation and image saving complete.")
