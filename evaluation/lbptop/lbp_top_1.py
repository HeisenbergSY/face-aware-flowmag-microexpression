import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

def process_videos(video_dir):
    """
    Process videos stored in folders within a single directory.
    Extract LBP-TOP features (5×5 blocks), save as .npy, and plot histograms.
    :param video_dir: Directory containing subfolders with cropped video frames.
    """
    for folder in os.listdir(video_dir):
        folder_path = os.path.join(video_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Collect all frames from the folder
        frames = []
        for frame_file in sorted(os.listdir(folder_path)):
            frame_path = os.path.join(folder_path, frame_file)

            # Skip non-frame files
            if not frame_file.endswith(('.jpg', '.png')) or frame_file.startswith("lbp_top_"):
                continue
            
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            if frame is not None:
                frames.append(frame)

        # Skip if no valid frames were found
        if len(frames) == 0:
            print(f"No valid frames found in {folder_path}. Skipping...")
            continue

        # Ensure frames have consistent dimensions
        try:
            frames = np.stack(frames)  # Combine frames into a 3D array
        except ValueError as e:
            print(f"Error combining frames in {folder_path}: {e}. Skipping...")
            continue

        # Perform LBP-TOP feature extraction with 5×5 blocks
        features = lbp_top_block(frames, rx=1, ry=1, rt=4)

        # Save features in the same folder as .npy
        feature_file = os.path.join(folder_path, "lbp_top_features.npy")
        np.save(feature_file, features)
        print(f"Saved LBP-TOP features for {folder} in {feature_file}")

        # Save histogram as an image
        histogram_image_path = os.path.join(folder_path, "lbp_top_histogram.png")
        save_histogram_image(features, histogram_image_path)
        print(f"Saved histogram image for {folder} in {histogram_image_path}")


def lbp_top_block(video_frames, rx=1, ry=1, rt=4, num_points=8, radius=1, block_size=5):
    """
    Compute LBP-TOP features for each 5×5 block in a sequence of frames.
    :param video_frames: List of frames (grayscale).
    :param rx, ry, rt: Radii for XY, XT, YT planes.
    :param num_points: Number of sampling points for LBP.
    :param radius: Radius for LBP.
    :param block_size: Number of blocks along each dimension.
    :return: Concatenated LBP-TOP histograms for all blocks.
    """
    num_frames, height, width = video_frames.shape
    block_height, block_width = height // block_size, width // block_size
    all_histograms = []

    for by in range(block_size):
        for bx in range(block_size):
            # Extract the block
            block_frames = video_frames[:, by * block_height:(by + 1) * block_height,
                                           bx * block_width:(bx + 1) * block_width]
            # Compute LBP-TOP for the block
            block_histograms = lbp_top(block_frames, rx, ry, rt, num_points, radius)
            all_histograms.append(block_histograms)

    return np.concatenate(all_histograms)  # Combine histograms for all blocks


def lbp_top(video_frames, rx=1, ry=1, rt=4, num_points=8, radius=1):
    """
    Compute LBP-TOP features for a sequence of frames.
    :param video_frames: List of frames (grayscale).
    :param rx: Radius in the X-axis.
    :param ry: Radius in the Y-axis.
    :param rt: Radius in the temporal axis.
    :param num_points: Number of sampling points for LBP.
    :param radius: Radius for LBP.
    :return: Histogram concatenation of LBP features from XY, XT, and YT planes.
    """
    num_frames, height, width = video_frames.shape
    lbp_xy_hist, lbp_xt_hist, lbp_yt_hist = [], [], []

    for t in range(rt, num_frames - rt):
        # XY plane
        lbp_xy = local_binary_pattern(video_frames[t], num_points, radius, method='uniform')
        lbp_xy_hist.append(np.histogram(lbp_xy, bins=np.arange(0, num_points + 3), density=True)[0])

        # XT plane
        lbp_xt = local_binary_pattern(video_frames[:, :, width // 2], num_points, radius, method='uniform')
        lbp_xt_hist.append(np.histogram(lbp_xt, bins=np.arange(0, num_points + 3), density=True)[0])

        # YT plane
        lbp_yt = local_binary_pattern(video_frames[:, height // 2, :], num_points, radius, method='uniform')
        lbp_yt_hist.append(np.histogram(lbp_yt, bins=np.arange(0, num_points + 3), density=True)[0])

    # Concatenate histograms from all three planes
    lbp_top_hist = np.concatenate([np.mean(lbp_xy_hist, axis=0),
                                    np.mean(lbp_xt_hist, axis=0),
                                    np.mean(lbp_yt_hist, axis=0)])
    return lbp_top_hist


def save_histogram_image(histogram, save_path):
    """
    Save the histogram as an image.
    :param histogram: The LBP-TOP histogram array.
    :param save_path: Path to save the histogram image.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(histogram)), histogram, width=0.8, color='blue')
    plt.title("LBP-TOP Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path)
    plt.close()


# Example usage
if __name__ == "__main__":
    video_dir = r"C:\Users\thepr\Desktop\Master\flowmag\inference\output_casme2_cropped"  # Replace with your directory path
    process_videos(video_dir)
