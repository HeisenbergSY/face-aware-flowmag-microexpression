import subprocess
import sys

# Ensure tqdm is installed
try:
    import tqdm
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    import tqdm  # Try importing again after installation
import os
import subprocess

# Set paths
DATA_DIR = r"C:\Users\thepr\Desktop\Master\data\casme2_test"  # Change this if your dataset is in a different lcdocation
OUTPUT_DIR = "./output_casme2"
CONFIG_PATH = "configs/alpha16.color10.yaml"
CHECKPOINT_PATH = "./checkpoints/raft_chkpt_00140.pth"
ALPHA = "20"  # Magnification factor
TTA_EPOCHS = "3"  # Number of epochs for test-time adaptation

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Iterate over each subject and sequence
for subject in os.listdir(DATA_DIR):
    subject_path = os.path.join(DATA_DIR, subject)
    if not os.path.isdir(subject_path):
        continue  # Skip files

    for sequence in sorted(os.listdir(subject_path)):  # Ensure order consistency
        sequence_path = os.path.join(subject_path, sequence)
        
        if not os.path.isdir(sequence_path):
            continue  # Skip files
        
        # Ensure directory is not empty
        if not os.listdir(sequence_path):
            print(f"Skipping empty sequence: {sequence_path}")
            continue  # Skip empty folders

        save_name = os.path.join(OUTPUT_DIR, f"{subject}_{sequence}_x{ALPHA}.mp4")  # Fix incorrect path issue
        
        command = [
            "C:/Users/thepr/Desktop/Master/master/Scripts/python.exe", "inference.py",
            "--config", CONFIG_PATH,
            "--frames_dir", sequence_path,
            "--resume", CHECKPOINT_PATH,
            "--save_name", save_name,
            "--alpha", ALPHA,
            "--test_time_adapt",
            "--tta_epoch", TTA_EPOCHS,
            "--output_video"
        ]
        
        print(f"Processing: {sequence_path}")
        print(f"Checking sequence path: {sequence_path}")
        print(f"Using checkpoint: {CHECKPOINT_PATH}")
        print(f"Executing command: {' '.join(command)}")
        print(f"Final output path: {save_name}")
        print(f"Available sequences: {os.listdir(subject_path)}")

        subprocess.run(command)