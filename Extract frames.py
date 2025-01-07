import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Extract frames from a video at regular intervals.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save extracted frames.
        frame_interval (int): Interval between frames to extract.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame every `frame_interval` frames
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frames:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1

        count += 1

    cap.release()
    print(f"Extracted {saved_frames} frames from {video_path}.")

# Example usage
video_path = "data/train/real/video1.mp4"
output_dir = "data/train/real/frames_video1"
extract_frames(video_path, output_dir)


