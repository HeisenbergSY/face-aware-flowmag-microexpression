# This is a modified script from Faraz to work with the cropped images from the dataset with UNIFORM BIN
import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from multiprocessing import Pool

def process_videos(video_dir):
    """
    Process videos stored in 'subXX/EPXX_XXX' folders.
    Extract LBP-TOP features (5×5 blocks), save as .npy, and plot histograms.
    :param video_dir: Root directory containing subXX folders with EPXX_XXX subfolders of frames.
    """
    for subfolder in os.listdir(video_dir):
        subfolder_path = os.path.join(video_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping {subfolder_path}: Not a directory.")
            continue

        print(f"Processing subfolder: {subfolder_path}")

        # Process EPXX_XXX folders inside subXX
        for episode_folder in os.listdir(subfolder_path):
            episode_path = os.path.join(subfolder_path, episode_folder)
            if not os.path.isdir(episode_path):
                print(f"Skipping {episode_path}: Not a directory.")
                continue

            print(f"Processing episode folder: {episode_path}")
            process_episode(episode_path)

def process_episode(episode_path):
    """
    Process a single episode folder (EPXX_XXX).
    Extract LBP-TOP features and save the results.
    :param episode_path: Path to the folder containing frames of a single episode.
    """
    frames = []
    first_shape = None  # Track the shape of the first valid image

    for frame_file in sorted(os.listdir(episode_path)):
        frame_path = os.path.join(episode_path, frame_file)

        # Skip non-image files and `lbp_top_histogram.png`
        if not os.path.isfile(frame_path) or not frame_file.lower().endswith(('.jpg', '.png')) or frame_file == "lbp_top_histogram.png":
            continue

        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if frame is None:
            continue  # Skip unreadable images

        if first_shape is None:
            first_shape = frame.shape  # Set first valid image shape

        # Ensure image has the same shape as the first valid frame
        if frame.shape != first_shape:
            print(f"Skipping {frame_path}: Shape mismatch {frame.shape} != {first_shape}")
            continue

        frames.append(frame)

    if len(frames) == 0:
        print(f"No valid frames found in {episode_path}. Skipping...")
        return

    try:
        frames = np.stack(frames)  # Combine frames into a 3D array
    except ValueError as e:
        print(f"Error combining frames in {episode_path}: {e}. Skipping...")
        return

    features = lbp_top_block(frames, rx=1, ry=1, rt=4)

    # Save features in the same folder as .npy
    feature_file = os.path.join(episode_path, "lbp_top_features.npy")
    np.save(feature_file, features)
    print(f"Saved LBP-TOP features for {episode_path} in {feature_file}")

    # Save histogram as an image
    histogram_image_path = os.path.join(episode_path, "lbp_top_histogram.png")
    save_histogram_image(features, histogram_image_path)
    print(f"Saved histogram image for {episode_path} in {histogram_image_path}")


    features = lbp_top_block(frames, rx=1, ry=1, rt=4)

    # Save features in the same folder as .npy
    feature_file = os.path.join(episode_path, "lbp_top_features.npy")
    np.save(feature_file, features)
    print(f"Saved LBP-TOP features for {episode_path} in {feature_file}")

    # Save histogram as an image
    histogram_image_path = os.path.join(episode_path, "lbp_top_histogram.png")
    save_histogram_image(features, histogram_image_path)
    print(f"Saved histogram image for {episode_path} in {histogram_image_path}")

def lbp_top_block(video_frames, rx=1, ry=1, rt=4, num_points=8, radius=1, block_size=5):
    """
    Compute LBP-TOP features for each 5×5 block in a sequence of frames.
    """
    num_frames, height, width = video_frames.shape
    block_height, block_width = height // block_size, width // block_size
    all_histograms = []

    for by in range(block_size):
        for bx in range(block_size):
            block_frames = video_frames[:, by * block_height:(by + 1) * block_height,
                                           bx * block_width:(bx + 1) * block_width]
            block_histograms = lbp_top(block_frames, rx, ry, rt, num_points, radius)
            all_histograms.append(block_histograms)

    return np.concatenate(all_histograms)

def get_uniform_lbp_mapping():
    """
    Creates a lookup table that maps 256-bin LBP values to 59-bin uniform LBP values
    based on MATLAB's UniformLBP8.txt mapping.
    """
    mapping = np.zeros(256, dtype=int)  # Initialize mapping table (default all to 0)
    
    bin_index = 0  # Start assigning uniform bins

    for i in range(256):
        binary_str = format(i, '08b')  # Convert number to 8-bit binary
        transitions = sum((binary_str[j] != binary_str[(j + 1) % 8]) for j in range(8))  # Count transitions

        if transitions <= 2:  # If uniform pattern (≤ 2 transitions)
            mapping[i] = bin_index  # Assign to next available uniform bin
            bin_index += 1
        else:
            mapping[i] = 58  # Assign all non-uniform patterns to bin 58 (MATLAB uses 59 bins, but Python is 0-indexed)
    
    return mapping

def lbp_top(video_frames, rx=1, ry=1, rt=4, num_points=8, radius=1):
    """
    Compute LBP-TOP features for a sequence of frames using MATLAB's uniform LBP mapping.
    """
    num_frames, height, width = video_frames.shape
    lbp_xy_hist, lbp_xt_hist, lbp_yt_hist = [], [], []

    mapping = get_uniform_lbp_mapping()  # Load the MATLAB-style mapping

    for t in range(rt, num_frames - rt):
        # Compute LBP in XY plane
        lbp_xy = local_binary_pattern(video_frames[t], num_points, radius, method='default')
        lbp_xy_mapped = np.vectorize(lambda x: mapping[int(x)])(lbp_xy)  # Apply mapping
        lbp_xy_hist.append(np.histogram(lbp_xy_mapped, bins=np.arange(0, 60), density=True)[0])

        # Compute LBP in XT plane
        lbp_xt = local_binary_pattern(video_frames[:, :, width // 2], num_points, radius, method='default')
        lbp_xt_mapped = np.vectorize(lambda x: mapping[int(x)])(lbp_xt)  # Apply mapping
        lbp_xt_hist.append(np.histogram(lbp_xt_mapped, bins=np.arange(0, 60), density=True)[0])

        # Compute LBP in YT plane
        lbp_yt = local_binary_pattern(video_frames[:, height // 2, :], num_points, radius, method='default')
        lbp_yt_mapped = np.vectorize(lambda x: mapping[int(x)])(lbp_yt)  # Apply mapping
        lbp_yt_hist.append(np.histogram(lbp_yt_mapped, bins=np.arange(0, 60), density=True)[0])

    # Concatenate histograms
    lbp_top_hist = np.concatenate([np.mean(lbp_xy_hist, axis=0),
                                    np.mean(lbp_xt_hist, axis=0),
                                    np.mean(lbp_yt_hist, axis=0)])
    return lbp_top_hist

def save_histogram_image(histogram, save_path):
    """
    Save the histogram as an image.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(histogram)), histogram, width=0.8, color='blue')
    plt.title("LBP-TOP Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    video_dir = r"C:\Users\thepr\Desktop\Master\flowmag\Cropped\Cropped"
    process_videos(video_dir)
