import os
import numpy as np
from scipy.signal import stft, get_window
from scipy.ndimage import zoom, gaussian_filter
from PIL import Image
from skimage import exposure

# Parameters
TARGET_SHAPE = (224, 224)  # Target shape for images
MAX_FREQ = 50  # Max frequency to keep in STFT
FS = 173.61  # Sampling frequency for EEG data


def load_eeg_data(file_path):
    """Load EEG data from a .txt file."""
    return np.loadtxt(file_path)


def apply_stft(eeg_data, fs=FS, window_function='hann'):
    """Apply Short-Time Fourier Transform (STFT) to EEG data using a specified window function."""
    nperseg = min(256, len(eeg_data))  # Adjusted window size for better frequency resolution
    noverlap = nperseg // 2  # 50% overlap for smoother transitions

    # Use get_window to generate the desired window function
    window = get_window(window_function, nperseg)

    f, t, Zxx = stft(eeg_data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, np.abs(Zxx)


def scale_to_grayscale(image_data):
    """Normalize the image data and convert it to grayscale using logarithmic scaling."""
    log_scaled_data = np.log1p(image_data)  # Logarithmic scaling to manage the dynamic range effectively
    normalized_data = (log_scaled_data - np.min(log_scaled_data)) / (np.max(log_scaled_data) - np.min(log_scaled_data))
    grayscale_data = (normalized_data * 255).astype(np.uint8)
    return grayscale_data


def adaptive_equalization(image_data):
    """Apply adaptive histogram equalization to the image data for enhanced contrast."""
    img_adapteq = exposure.equalize_adapthist(image_data, clip_limit=0.03)  # CLAHE for better visual contrast
    return (img_adapteq * 255).astype(np.uint8)


def smooth_image(image_data, sigma=2):
    """Apply Gaussian smoothing to the image data for noise reduction."""
    return gaussian_filter(image_data, sigma=sigma)


def create_stft_image(eeg_data, target_shape=TARGET_SHAPE, max_freq=MAX_FREQ, fs=FS, window_function='hann'):
    """Create a smoothed grayscale image from EEG data using STFT, with focus on meaningful frequency range."""
    f, stft_data = apply_stft(eeg_data, fs=fs, window_function=window_function)
    freq_idx = f <= max_freq  # Focus on relevant frequencies (up to max_freq Hz)
    stft_data = stft_data[freq_idx, :]
    # Zoom to fit the target shape
    scaled_stft = zoom(stft_data, (target_shape[0] / stft_data.shape[0], target_shape[1] / stft_data.shape[1]), order=1)
    # Convert to grayscale and apply adaptive equalization for contrast enhancement
    grayscale_image = scale_to_grayscale(scaled_stft)
    equalized_image = adaptive_equalization(grayscale_image)
    # Apply Gaussian smoothing for a smoother appearance
    smoothed_image = smooth_image(equalized_image, sigma=2)
    return smoothed_image


def process_and_save_stft_images(input_file, output_folder, target_shape=TARGET_SHAPE, window_function='hann'):
    """Process EEG files to create and save STFT-based smoothed grayscale images."""
    # Load EEG data from the text file
    eeg_data = load_eeg_data(input_file)

    # Create an STFT image
    stft_image = create_stft_image(eeg_data, target_shape, window_function=window_function)
    eeg_pil = Image.fromarray(stft_image)

    # Define output filename based on input file
    base_filename = os.path.basename(input_file).replace('.txt', '')
    image_filename = f'{base_filename}.png'

    # Save the STFT image
    eeg_pil.save(os.path.join(output_folder, image_filename))

    print(f'Saved STFT image for {base_filename} in {output_folder}')


# Define the input and output directories
input_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Preprocessed data/Preprocessing 1'
output_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/EEG_Dimension Transformation Images/Short-Time Fourier Transform (STFT)'

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
            process_and_save_stft_images(input_file_path, output_dir, window_function='hann')

print("STFT transformation and image saving complete.")
