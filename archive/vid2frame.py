import os
import cv2

def extract_frames_from_videos(videos_dir):
    # Iterate through all subfolders in the videos directory
    for subdir, _, files in os.walk(videos_dir):
        for file in files:
            if file.endswith(('.mp4')):  # Add video file extensions as needed
                video_path = os.path.join(subdir, file)
                output_dir = subdir  # Save frames in the same directory as the video

                # Create VideoCapture object
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Processing {file} with {frame_count} frames...")

                frame_number = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Save each frame as a JPEG image
                    output_path = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
                    cv2.imwrite(output_path, frame)
                    frame_number += 1

                cap.release()
                print(f"Frames from {file} saved to {output_dir}.")

# Example usage
videos_dir = r"C:\Users\thepr\Desktop\Master\flowmag\results\03_25_2025-20-13-34-alpha16.color10.raft\inference"  # Replace with the path to your folder
extract_frames_from_videos(videos_dir)
