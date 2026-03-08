# delete old npy and histogram files
import os

def delete_old_files(root_dir):
    """
    Traverse directories and delete files named 'lbp_top_features.npy' and 'lbp_top_histogram.png'.

    :param root_dir: Root directory containing subdirectories with files to delete.
    """
    # Loop through all subdirectories and files
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file in ["lbp_top_features.npy", "lbp_top_histogram.png"]:
                file_path = os.path.join(subdir, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    # Specify the root directory to start the deletion process
    root_directory = r"C:\Users\thepr\Desktop\Master\flowmag\inference\output_casme2_cropped"
    delete_old_files(root_directory)