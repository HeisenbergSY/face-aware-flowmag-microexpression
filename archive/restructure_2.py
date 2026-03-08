import os
import shutil

def restructure_folders(root_dir):
    """
    Restructure folders by moving `EPXX_XXX` folders out of their parent `subXX` directories,
    and rename them by prefixing the parent directory name.
    :param root_dir: The root directory containing `subXX` folders.
    """
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)

        # Check if the subfolder is a directory
        if not os.path.isdir(subfolder_path):
            print(f"Skipping {subfolder_path}: Not a directory.")
            continue

        print(f"Processing folder: {subfolder_path}")

        # Iterate through the contents of the subXX folder
        for episode_folder in os.listdir(subfolder_path):
            episode_path = os.path.join(subfolder_path, episode_folder)

            # Check if the item is a directory (EPXX_XXX folder)
            if not os.path.isdir(episode_path):
                print(f"Skipping {episode_path}: Not a directory.")
                continue

            # Construct the new folder name and path
            new_folder_name = f"{subfolder}_{episode_folder}"
            new_folder_path = os.path.join(root_dir, new_folder_name)

            # Move and rename the folder
            try:
                shutil.move(episode_path, new_folder_path)
                print(f"Moved and renamed: {episode_path} -> {new_folder_path}")
            except Exception as e:
                print(f"Failed to move {episode_path}: {e}")

        # Remove the now-empty `subXX` folder
        try:
            os.rmdir(subfolder_path)
            print(f"Removed empty folder: {subfolder_path}")
        except OSError as e:
            print(f"Failed to remove {subfolder_path}: {e}")

if __name__ == "__main__":
    root_directory = r"C:\Users\thepr\Desktop\Master\flowmag\results\03_23_2025-11-12-49-alpha16.color10.raft\inference"  # Replace with your root directory path
    restructure_folders(root_directory)
