import cv2
import os

def extract_frames(video_path, base_output_dir, frame_interval=30):
    """
    Extract frames from a video at regular intervals and save them in a separate folder.

    Args:
        video_path (str): Path to the video file.
        base_output_dir (str): Base directory to save extracted frames.
        frame_interval (int): Interval between frames to extract.
    """
    # Extract video name and create a unique subfolder
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(base_output_dir, f"frames_{video_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    print(f"Processing video: {video_path}")
    count = 0
    saved_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or no frames to read.")
            break

        # Save every `frame_interval` frame
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frames:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1

        count += 1

    cap.release()
    print(f"Extracted {saved_frames} frames from {video_path} to {output_dir}.")

def batch_process_videos(video_dir, base_output_dir, frame_interval=30):
    """
    Batch process all videos in a directory to extract frames.

    Args:
        video_dir (str): Directory containing video files.
        base_output_dir (str): Base directory to save extracted frames.
        frame_interval (int): Interval between frames to extract.
    """
    # Loop through all videos in the directory
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        if os.path.isfile(video_path) and video_file.endswith(('.mp4', '.avi', '.mkv')):
            extract_frames(video_path, base_output_dir, frame_interval)

# Example usage
video_dir = "C:/Users/Vikram/DFDC/data/train/real"  # Directory containing video files
base_output_dir = "C:/Users/Vikram/DFDC/data/processed/train/real"  # Directory to save extracted frames
frame_interval = 30  # Extract one frame every 30 frames

batch_process_videos(video_dir, base_output_dir, frame_interval)
