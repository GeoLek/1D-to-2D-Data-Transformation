import os
import random
import shutil

# Define paths
base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Augmented EEG Dimension Transformation Images/Short-Time Fourier Transform (STFT)'  # Update with the path to your augmented images
output_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/EEG Splitted Dataset_Final/Short-Time Fourier Transform (STFT)'  # Output directory for the train, validation, and test splits

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Ensure output directories for train, val, and test sets
splits = ['train', 'val', 'test']
sets = ['Set A', 'Set B', 'Set C', 'Set D', 'Set E']
letters = ['Z', 'O', 'N', 'F', 'S']

for split in splits:
    for set_name, letter in zip(sets, letters):
        os.makedirs(os.path.join(output_base_dir, split, set_name, letter), exist_ok=True)

# Function to split images into train, val, and test
def split_images(image_list, train_ratio, val_ratio, test_ratio):
    random.shuffle(image_list)
    total_images = len(image_list)
    train_split = int(total_images * train_ratio)
    val_split = int(total_images * (train_ratio + val_ratio))

    train_images = image_list[:train_split]
    val_images = image_list[train_split:val_split]
    test_images = image_list[val_split:]

    return train_images, val_images, test_images

# Split each category's images
for set_name, letter in zip(sets, letters):
    input_folder = os.path.join(base_dir, set_name, letter)
    images = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    # Split images into train, validation, and test
    train_images, val_images, test_images = split_images(images, train_ratio, val_ratio, test_ratio)

    # Move images to corresponding directories
    for image_name in train_images:
        shutil.copy(os.path.join(input_folder, image_name), os.path.join(output_base_dir, 'train', set_name, letter, image_name))

    for image_name in val_images:
        shutil.copy(os.path.join(input_folder, image_name), os.path.join(output_base_dir, 'val', set_name, letter, image_name))

    for image_name in test_images:
        shutil.copy(os.path.join(input_folder, image_name), os.path.join(output_base_dir, 'test', set_name, letter, image_name))

    print(f"Completed split for {set_name}/{letter}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

print("Dataset splitting complete.")
