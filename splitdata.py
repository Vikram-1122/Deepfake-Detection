import os
import shutil
import random

def split_data(source_dir, dest_dir, split_ratios=(0.8, 0.1, 0.1)):
    """
    Split data into train, validation, and test sets.

    Args:
        source_dir (str): Directory containing data (e.g., 'real', 'fake').
        dest_dir (str): Base directory for train/val/test splits.
        split_ratios (tuple): Ratios for train, val, and test splits.
    """
    # Collect files from all subdirectories
    files = []
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path):
                    files.append(file_path)
        else:
            print(f"Skipping non-directory: {subdir_path}")

    if not files:
        print(f"No files found in {source_dir}.")
        return

    # Shuffle files for randomness
    random.shuffle(files)

    # Calculate split indices
    train_split = int(len(files) * split_ratios[0])
    val_split = train_split + int(len(files) * split_ratios[1])

    # Split files into train, val, test
    train_files = files[:train_split]
    val_files = files[train_split:val_split]
    test_files = files[val_split:]

    # Copy files to train, val, test directories
    for split, split_files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        split_dir = os.path.join(dest_dir, split, os.path.basename(source_dir))
        os.makedirs(split_dir, exist_ok=True)

        for file_path in split_files:
            src = file_path
            dst = os.path.join(split_dir, os.path.basename(file_path))
            try:
                shutil.copy(src, dst)
            except PermissionError:
                print(f"Skipping file due to permission error: {src}")
            except Exception as e:
                print(f"Error copying {src}: {e}")

# Example usage
source_dir_real = "C:/Users/Vikram/DFDC/data/processed/train/real/normalized"
dest_dir = "C:/Users/Vikram/DFDC/data/final"

split_data(source_dir_real, dest_dir)
