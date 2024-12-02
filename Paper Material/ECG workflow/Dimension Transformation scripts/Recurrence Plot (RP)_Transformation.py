import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom


def load_ecg_data(file_path):
    """Load ECG data from a CSV file."""
    return pd.read_csv(file_path)['MLII'].values


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
    """Generate a recurrence plot from 1D ECG data by comparing all pairs of points."""
    data = data.reshape(-1, 1)
    # Compute the recurrence matrix based on the epsilon threshold
    distances = np.abs(data - data.T)
    recurrence_matrix = (distances < epsilon).astype(np.uint8)
    return recurrence_matrix


def create_and_save_rp_images(record_number, input_file, output_folder, target_shape=(224, 224)):
    """Process ECG files to create and save recurrence plot images."""
    os.makedirs(output_folder, exist_ok=True)

    # Load the beat segments from the CSV file
    beats_df = pd.read_csv(input_file)
    symbols = beats_df['Symbol'].tolist()
    beat_segments = beats_df.drop(columns=['Symbol']).values

    # Create output directory for the record
    record_dir = os.path.join(output_folder, f'record_{record_number:03d}')
    os.makedirs(record_dir, exist_ok=True)

    # Process and save each beat segment as an image
    for i, (beat_segment, symbol) in enumerate(zip(beat_segments, symbols)):
        # Normalize the beat segment
        beat_segment = (beat_segment - np.mean(beat_segment)) / np.std(beat_segment)

        # Select threshold epsilon based on a fixed recurrence rate
        epsilon = select_threshold(beat_segment, recurrence_rate=0.05)

        # Generate the recurrence plot matrix
        rp_matrix = create_recurrence_plot(beat_segment, epsilon)

        # Resize the recurrence matrix to the target image size
        resized_rp = zoom(rp_matrix, (target_shape[0] / rp_matrix.shape[0],
                                      target_shape[1] / rp_matrix.shape[1]), order=0)
        rp_image = Image.fromarray(resized_rp * 255).convert('L')  # Convert to grayscale image

        # Sanitize the symbol to create a valid file name
        sanitized_symbol = "".join(c if c.isalnum() else '_' for c in symbol)
        # Save the plot as a PNG file with the annotation in the file name
        image_filename = f'beat_{i + 1}_symbol_{sanitized_symbol}.png'
        rp_image.save(os.path.join(record_dir, image_filename))

    print(f'Saved images for record {record_number:03d} in {record_dir}')


# Define the directory containing the output CSV files
input_dir = '/home/orion/Geo/Projects/1D-to-2D-Data-Transformation/Output_updated/ECG/Extracted_Processed_Beats'
output_dir = '/home/orion/Geo/Projects/1D-to-2D-Data-Transformation/Output_updated/ECG/Recurrence Plots'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of record numbers (e.g., 100, 101, 102, ...)
record_numbers = list(range(100, 235))

# Loop through each record number and process the files
for record_number in record_numbers:
    csv_filename = os.path.join(input_dir, f'{record_number:03d}_beats.csv')
    if os.path.exists(csv_filename):
        create_and_save_rp_images(record_number, csv_filename, output_dir)
    else:
        print(f'CSV file for record {record_number:03d} not found. Skipping...')

print("Processing complete.")
