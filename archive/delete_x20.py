import os

# Directory where the folders are located
folder_path = r'C:\Users\thepr\Desktop\Master\flowmag\inference\output_casme2_cropped'  # Change this to the path where your folders are located

# Walk through all directories and subdirectories
for root, dirs, files in os.walk(folder_path):
    for folder_name in dirs:
        # Check if the folder name ends with '_x20'
        if folder_name.endswith('_x20'):
            # Create the new folder name by removing the '_x20' suffix
            new_folder_name = folder_name[:-4]  # Remove the last 4 characters ('_x20')
            
            # Get the full path of the old and new folder names
            old_folder_path = os.path.join(root, folder_name)
            new_folder_path = os.path.join(root, new_folder_name)
            
            # Rename the folder
            os.rename(old_folder_path, new_folder_path)
            print(f'Renamed "{folder_name}" to "{new_folder_name}"')