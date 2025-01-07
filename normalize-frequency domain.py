from PIL import Image
import os

def normalize_image(image_path, output_path, size=(224, 224)):
    """
    Resize an image to a fixed size.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the resized image.
        size (tuple): Target size (width, height).
    """
    img = Image.open(image_path)
    img_resized = img.resize(size)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_resized.save(output_path)

def process_frames_to_normalized(input_dir, output_dir, size=(224, 224)):
    """
    Normalize all frames in a directory to a fixed size.

    Args:
        input_dir (str): Directory containing frequency-domain images.
        output_dir (str): Directory to save normalized images.
        size (tuple): Target size (width, height).
    """
    for video_folder in os.listdir(input_dir):
        video_freq_dir = os.path.join(input_dir, video_folder)
        video_norm_dir = os.path.join(output_dir, video_folder)

        if not os.path.isdir(video_freq_dir):
            print(f"Skipping {video_folder}: Not a directory.")
            continue

        for freq_file in os.listdir(video_freq_dir):
            freq_path = os.path.join(video_freq_dir, freq_file)
            norm_path = os.path.join(video_norm_dir, freq_file)
            normalize_image(freq_path, norm_path, size)

# Example usage
input_dir = "C:/Users/Vikram/DFDC/data/processed/train/real/frequency"
output_dir = "C:/Users/Vikram/DFDC/data/processed/train/real/normalized"
process_frames_to_normalized(input_dir, output_dir)
