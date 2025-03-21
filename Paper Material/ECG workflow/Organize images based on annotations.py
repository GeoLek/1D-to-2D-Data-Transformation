import os
import shutil

# Base directory containing folders like "record_100", "record_101", etc.
data_dir = ''
# Destination directory to organize images
output_dir = os.path.join(data_dir, 'Organized')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

##############################################################################
# Extended Annotation Mapping
##############################################################################
# Each final class (N, S, V, F, Q) corresponds to multiple symbol variations.
annotation_equiv = {
    'N': ['N', 'L', 'R', 'e', 'j'],
    'S': ['A', 'a', 'J', 'S'],
    'V': ['V', 'E'],
    'F': ['F'],
    'Q': ['/', 'f', 'Q']
}

# Create subfolders for each final class
for final_class in annotation_equiv:
    os.makedirs(os.path.join(output_dir, final_class), exist_ok=True)

# Iterate over subdirectories named "record_XXX" within data_dir
for record_folder in sorted(os.listdir(data_dir)):
    record_path = os.path.join(data_dir, record_folder)

    # Skip non-directory items or the "Organized" folder
    if not os.path.isdir(record_path) or record_folder == 'Organized':
        continue

    # Process PNG files inside each record_XXX folder
    for file_name in sorted(os.listdir(record_path)):
        if file_name.endswith('.png'):
            source_path = os.path.join(record_path, file_name)

            # Split filename into base and extension to ensure strict matching
            base_name, ext = os.path.splitext(file_name)
            # We already know ext == '.png' from the check above

            # Attempt to match the base name with each final_class symbol
            copied = False
            for final_class, symbol_list in annotation_equiv.items():
                for symbol in symbol_list:
                    # We check if the base_name ends exactly with "symbol_{symbol}"
                    # Example: base_name might be "beat_3_symbol_N"
                    # We require the suffix to be: "_symbol_N"
                    suffix = f'symbol_{symbol}'
                    if base_name.endswith(suffix):
                        # Construct a new file name to avoid overwriting
                        new_file_name = f'{record_folder}_{base_name}{ext}'
                        dest_path = os.path.join(output_dir, final_class, new_file_name)
                        shutil.copy(source_path, dest_path)
                        copied = True
                        break
                if copied:
                    break

print("Organized files into annotation folders successfully!")
