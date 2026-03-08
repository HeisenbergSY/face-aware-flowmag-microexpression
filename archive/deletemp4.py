import os

def rename_folders(directory):
    """
    Rename folders in the given directory by removing the .mp4 extension at the end.
    :param directory: Path to the parent directory containing the folders.
    """
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        
        # Check if it's a folder and ends with '.mp4'
        if os.path.isdir(folder_path) and folder.endswith('.mp4'):
            # New folder name without '.mp4'
            new_folder_name = folder[:-4]
            new_folder_path = os.path.join(directory, new_folder_name)
            
            # Rename the folder
            os.rename(folder_path, new_folder_path)
            print(f"Renamed: {folder} -> {new_folder_name}")
        else:
            print(f"Skipped: {folder}")

# Example usage
directory = r"C:\Users\thepr\Desktop\Master\flowmag\inference\output_casme2_cropped"  # Replace with your directory
rename_folders(directory)
