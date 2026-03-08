import os

# Directory where the folders are located
folder_path = r'C:\Users\thepr\Desktop\Master\flowmag\inference\output_casme2_cropped'  # Change this to the path where your folders are located

# Walk through all subfolders and files in the directory
for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        # Check if the file is an .mp4 file
        if file_name.endswith('.mp4'):
            # Get the full path of the file
            file_path = os.path.join(root, file_name)
            
            # Delete the .mp4 file
            os.remove(file_path)
            print(f'Deleted: {file_path}')
