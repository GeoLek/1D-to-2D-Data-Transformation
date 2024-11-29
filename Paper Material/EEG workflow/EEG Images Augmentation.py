import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from PIL import Image, ImageEnhance, ImageFilter
import random

# Define paths
base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/EEG_Dimension Transformation Images/Short-Time Fourier Transform (STFT)'  # Update this path
output_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Augmented EEG Dimension Transformation Images/Short-Time Fourier Transform (STFT)'  # Update this path
sets = ['Set A', 'Set B', 'Set C', 'Set D', 'Set E']
letters = ['Z', 'O', 'N', 'F', 'S']
target_images_count = 10000  # Target number of images per folder

# Very slight augmentation settings
datagen = ImageDataGenerator(
    rotation_range=3,  # Very minimal rotation
    zoom_range=0.02,  # Very slight zoom in and out
    brightness_range=[0.95, 1.05],  # Slight brightness adjustment
    fill_mode='nearest'  # Fill mode for edge artifacts
)


def apply_custom_augmentations(image):
    """Apply additional augmentations for contrast and very minimal blur."""
    # Slight contrast adjustment
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(np.random.uniform(0.98, 1.02))  # Very minimal contrast adjustment

    # Extremely minimal blur
    image = image.filter(ImageFilter.GaussianBlur(0.1))  # Minimal blur for subtle effect

    return image


# Loop through each set and letter directory
for set_name, letter in zip(sets, letters):
    input_folder = os.path.join(base_dir, set_name, letter)
    output_folder = os.path.join(output_dir, set_name, letter)
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of all images in the folder
    images = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    current_count = len(images)

    print(f"Processing {set_name}/{letter}: {current_count} images found.")

    if current_count >= target_images_count:
        print(f"Skipping {set_name}/{letter} as it already has {current_count} images.")
        continue

    # Calculate how many augmentations to perform per image
    augmentations_per_image = target_images_count // current_count

    # Loop through each image in the folder
    for image_name in images:
        # Load the image
        img_path = os.path.join(input_folder, image_name)
        img = load_img(img_path)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Perform a fixed number of augmentations for each image
        for i in range(augmentations_per_image):
            # Generate augmented image using ImageDataGenerator
            for batch in datagen.flow(img_array, batch_size=1):
                augmented_img = array_to_img(batch[0])

                # Apply additional light augmentations
                augmented_img = apply_custom_augmentations(augmented_img)

                # Save the augmented image
                new_image_name = f"{os.path.splitext(image_name)[0]}_aug_{i}.png"
                augmented_img.save(os.path.join(output_folder, new_image_name))

                break  # Exit inner loop after one augmentation to control the count

    print(f"Completed {set_name}/{letter}. Total images generated: {target_images_count}")

print("Image augmentation complete.")