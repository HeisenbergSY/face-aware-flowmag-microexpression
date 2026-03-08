# this script is used to move the results of the experiment to a new directory structure
# the original structure is: results/sub01_EP02_01f and the new structure is: results/sub01/EP02_01f
import os
import shutil

# Path to your results folder
results_path = r"C:\Users\thepr\Desktop\Master\flowmag\inference\output_casme2_cropped"  # Change this to your actual path

# Iterate through the folders in the result directory
for folder_name in os.listdir(results_path):
    full_path = os.path.join(results_path, folder_name)

    if not os.path.isdir(full_path):
        continue  # Skip files

    # Split the folder name (e.g., sub01_EP02_01f)
    if "_" not in folder_name:
        print(f"Skipping {folder_name}, not matching expected pattern.")
        continue

    subject, episode = folder_name.split("_", 1)
    subject_dir = os.path.join(results_path, subject)
    new_episode_path = os.path.join(subject_dir, episode)

    # Create subject directory if it doesn't exist
    os.makedirs(subject_dir, exist_ok=True)

    # Move the folder
    shutil.move(full_path, new_episode_path)
    print(f"Moved {folder_name} → {subject}/{episode}")
