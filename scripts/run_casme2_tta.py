import os
import subprocess
import multiprocessing

# Paths
DATASET_PATH = r"C:\Users\thepr\Desktop\Master\CASME II\Cropped\Cropped"  # Root folder containing CASME II sequences
OUTPUT_PATH = "inference"  # Where results will be saved
CHECKPOINT_PATH = "./checkpoints/raft_chkpt_00140.pth"

# Get available CPU cores
num_workers = min(8, multiprocessing.cpu_count() // 2)

PYTHON_EXECUTABLE = "C:/Users/thepr/Desktop/Master/master/Scripts/python.exe"

# Iterate through each subject and sequence
for subject in os.listdir(DATASET_PATH):
    subject_path = os.path.join(DATASET_PATH, subject)
    
    if not os.path.isdir(subject_path):
        continue  # Skip files, process only folders

    print(f"📂 Processing Subject: {subject}")

    for sequence in os.listdir(subject_path):
        sequence_path = os.path.join(subject_path, sequence)

        if not os.path.isdir(sequence_path):
            print(f"❌ Skipping (Not a folder): {sequence_path}")
            continue  # Skip non-folder files

        # Check if PNG/JPG frames exist
        frame_list = [f for f in os.listdir(sequence_path) if f.endswith('.png') or f.endswith('.jpg')]
        if len(frame_list) == 0:
            print(f"⚠️ No PNG/JPG frames found in {sequence_path}, skipping...")
            continue  # Skip empty folders

        num_frames = len(frame_list)  # Count frames

        print(f"🎬 Processing Sequence: {sequence} ({num_frames} frames)")

        # Define save path to match CASME II structure
        save_folder = os.path.join(OUTPUT_PATH, subject, sequence)
        os.makedirs(save_folder, exist_ok=True)

        # Construct the command
        command = [
            PYTHON_EXECUTABLE, "inference.py",
            "--config", "configs/alpha16.color10.yaml",
            "--frames_dir", sequence_path,  # Use correct sequence folder
            "--resume", CHECKPOINT_PATH,
            "--save_name", save_folder,  # Save in corresponding CASME II folder
            "--alpha", "20",
            "--test_time_adapt",
            "--tta_epoch", "10",  # Allows early stopping to choose the best epoch
            "--output_video"
        ]

        print(f"🚀 Running TTA on {sequence} ({num_frames} frames)")
        subprocess.run(command)
