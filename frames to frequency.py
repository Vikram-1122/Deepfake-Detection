import os
import numpy as np
from scipy.fftpack import fft2, fftshift
from PIL import Image

def convert_to_frequency(image_path, output_path):
    """
    Convert an image to its frequency domain representation using FFT.
    """
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    freq_transform = np.log(1 + np.abs(fftshift(fft2(img_array))))
    freq_transform = (freq_transform / freq_transform.max()) * 255

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    freq_image = Image.fromarray(freq_transform.astype(np.uint8))
    freq_image.save(output_path)

def process_frames_to_frequency(input_dir, output_dir):
    """
    Convert all frames in a directory to their frequency domain representation.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    for video_folder in os.listdir(input_dir):
        video_frame_dir = os.path.join(input_dir, video_folder)
        video_freq_dir = os.path.join(output_dir, video_folder)

        if not os.path.isdir(video_frame_dir):
            print(f"Skipping {video_folder}: Not a directory.")
            continue

        for frame_file in os.listdir(video_frame_dir):
            frame_path = os.path.join(video_frame_dir, frame_file)
            freq_path = os.path.join(video_freq_dir, frame_file)
            convert_to_frequency(frame_path, freq_path)

# Example usage
input_dir = "C:/Users/Vikram/DFDC/data/processed/train/real"
output_dir = "C:/Users/Vikram/DFDC/data/processed/train/real/frequency"
process_frames_to_frequency(input_dir, output_dir)
