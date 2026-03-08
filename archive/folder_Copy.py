import os
import shutil

def copy_folder_structure(src, dest):
    if not os.path.exists(src):
        print(f"Source directory '{src}' does not exist.")
        return
    
    for root, dirs, _ in os.walk(src):
        for dir_name in dirs:
            src_dir_path = os.path.join(root, dir_name)
            relative_path = os.path.relpath(src_dir_path, src)
            dest_dir_path = os.path.join(dest, relative_path)
            
            os.makedirs(dest_dir_path, exist_ok=True)
            print(f"Created: {dest_dir_path}")

if __name__ == "__main__":
    source_directory = input(r"C:\Users\thepr\Desktop\Master\flowmag\Cropped\Cropped")
    destination_directory = input(r"C:\Users\thepr\Desktop\Master\flowmag\After_TTA")
    
    copy_folder_structure(source_directory, destination_directory)
    print("Folder structure copied successfully.")
