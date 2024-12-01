import os
import shutil

# Define directories
data_dir = '/home/orion/Geo/Projects/1D-to-2D-Data-Transformation/Output_updated/ECG/Recurrence Plots'
output_dir = '/home/orion/Geo/Projects/1D-to-2D-Data-Transformation/Output_updated/ECG/Organized-RP'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Desired annotations and create subfolders
desired_annotations = ['N', 'S', 'V', 'F', 'Q']
for annotation in desired_annotations:
    os.makedirs(os.path.join(output_dir, annotation), exist_ok=True)

# Iterate through all folders in the data directory
for record_folder in os.listdir(data_dir):
    record_path = os.path.join(data_dir, record_folder)

    # Skip if not a directory
    if not os.path.isdir(record_path):
        continue

    # Process PNG images inside the folder
    for file_name in os.listdir(record_path):
        if file_name.endswith('.png'):
            # Identify the annotation in the file name
            for annotation in desired_annotations:
                if f'symbol_{annotation}' in file_name:
                    # Copy the file to the corresponding annotation folder
                    source_path = os.path.join(record_path, file_name)
                    destination_path = os.path.join(output_dir, annotation, file_name)
                    shutil.copy(source_path, destination_path)
                    break  # Stop checking other annotations for this file

print("Organized files into annotation folders successfully!")
