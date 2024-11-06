import os
import numpy as np
import matplotlib.pyplot as plt


def load_eeg_data(file_path):
    """Load EEG data from a text file."""
    return np.loadtxt(file_path)


def plot_eeg_signal(data, output_path):
    """Plot the EEG signal as a simple 2D line plot and save as an image."""
    plt.figure(figsize=(4, 4))  # Adjust the size as necessary for clarity
    plt.plot(data, color='black', linewidth=1)  # Plot the signal in black
    plt.axis('off')  # Hide axes for a clean look
    plt.tight_layout(pad=0)

    # Save the plot as a PNG image
    plt.savefig(output_path, format='png', dpi=224)  # High DPI for better resolution
    plt.close()


def process_and_save_images(set_label, record_name, input_file, output_folder):
    """Load EEG data, create a line plot, and save it as an image."""
    os.makedirs(output_folder, exist_ok=True)

    # Load EEG data
    eeg_data = load_eeg_data(input_file)

    # Define output file path for the image
    image_filename = f"{record_name}.png"
    output_path = os.path.join(output_folder, image_filename)

    # Plot and save the EEG signal as a 2D image
    plot_eeg_signal(eeg_data, output_path)
    print(f"Saved plot image for {record_name} in {output_folder}")


# Define base directories for input and output data
input_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Epilepsy EEG Dataset (University of Bonn)/Preprocessed data/Preprocessing 1'
output_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/EEG_Dimension Transformation Images/Simple Line Plot'

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
            process_and_save_images(set_name, record_name, input_file, output_dir)

print("Line plot creation and image saving complete.")
