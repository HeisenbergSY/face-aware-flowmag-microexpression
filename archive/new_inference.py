import os
import subprocess
from pathlib import Path
import sys

# === Configuration ===
DATA_DIR = "Cropped/Cropped"  # CASME II cropped faces
OUTPUT_DIR = "output_casme2_landmark"  # Where to save the magnified videos
CONFIG_PATH = "results/03_25_2025-20-13-34-alpha16.color10.raft/config.yaml"  # Your training config
CHECKPOINT_PATH = "results/03_25_2025-20-13-34-alpha16.color10.raft/checkpoints/chkpt_00050.pth"
ALPHA = "20"  # Magnification factor

# === Ensure output directory exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Function to find all valid frame directories ===
def find_frame_directories(data_dir):
    frame_dirs = []
    for root, _, files in os.walk(data_dir):
        # Only keep directories with image files
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            frame_dirs.append(root)
    return frame_dirs

# === Loop over directories ===
frame_dirs = find_frame_directories(DATA_DIR)

for frame_dir in sorted(frame_dirs):
    relative_path = os.path.relpath(frame_dir, DATA_DIR)
    save_name = relative_path.replace(os.sep, "_")

    command = [
        sys.executable, "inference.py",
        "--config", CONFIG_PATH,
        "--frames_dir", frame_dir,
        "--resume", CHECKPOINT_PATH,
        "--save_name", save_name,
        "--alpha", ALPHA,
        "--output_video"
    ]
    
    print(f"🔍 Running inference on: {frame_dir}")
    subprocess.run(command)

print("✅ Landmark-aware inference completed for all cropped sequences.")
