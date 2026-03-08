import os

# Directory where the folders are located
folder_path = r'C:\Users\thepr\Desktop\Master\flowmag\inference\output_casme2_cropped'  # Change this to the path where your folders are located

# Loop through all the folders in the directory
for folder_name in os.listdir(folder_path):
    # Check if the folder name starts with 'Cropped_'
    if folder_name.startswith('Cropped_'):
        # Create the new folder name by removing the 'Cropped_' prefix
        new_folder_name = folder_name.replace('Cropped_', '', 1)
        
        # Get the full path of the old and new folder names
        old_folder_path = os.path.join(folder_path, folder_name)
        new_folder_path = os.path.join(folder_path, new_folder_name)
        
        # Rename the folder
        os.rename(old_folder_path, new_folder_path)
        print(f'Renamed "{folder_name}" to "{new_folder_name}"')
