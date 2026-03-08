import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from multiprocessing import Pool


def process_episode(episode_path):
    """
    Process a single episode folder (EPXX_XXX).
    Extract LBP-TOP features and save the results.
    :param episode_path: Path to the folder containing frames of a single episode.
    """
    print(f"Processing episode folder: {episode_path}")

    # Collect all frames from the episode folder
    frames = []
    for frame_file in sorted(os.listdir(episode_path)):
        frame_path = os.path.join(episode_path, frame_file)

        # Skip invalid files and output files (e.g., histogram image)
        if not os.path.isfile(frame_path) or not frame_file.lower().endswith(('.jpg', '.png')):
            continue
        if frame_file == "lbp_top_histogram.png":  # Skip histogram image
            print(f"Skipping histogram file: {frame_file}")
            continue

        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if frame is not None:
            frames.append(frame)

    # Skip if no valid frames were found
    if len(frames) == 0:
        print(f"No valid frames found in {episode_path}. Skipping...")
        return

    # Ensure frames have consistent dimensions
    try:
        frames = np.stack(frames)  # Combine frames into a 3D array
    except ValueError as e:
        print(f"Error combining frames in {episode_path}: {e}. Skipping...")
        return

    # Perform LBP-TOP feature extraction with 5×5 blocks
    features = lbp_top_block(frames, rx=1, ry=1, rt=4)

    # Save features in the same folder as .npy
    feature_file = os.path.join(episode_path, "lbp_top_features.npy")
    np.save(feature_file, features)
    print(f"Saved LBP-TOP features for {episode_path} in {feature_file}")

    # Save histogram as an image
    histogram_image_path = os.path.join(episode_path, "lbp_top_histogram.png")
    save_histogram_image(features, histogram_image_path)
    print(f"Saved histogram image for {episode_path} in {histogram_image_path}")


def lbp_top_block(video_frames, rx=1, ry=1, rt=4, num_points=4, block_size=5):
    """
    Compute LBP-TOP features for each 5×5 block in a sequence of frames.
    Parameters match the paper specifications exactly.

    Changes made:
    1. Proper handling of temporal radius for XT and YT planes
    2. Improved normalization strategy
    3. Better handling of block boundaries
    4. Added volume-based computation for XT and YT planes
    """
    num_frames, height, width = video_frames.shape
    block_height = height // block_size
    block_width = width // block_size

    # Initialize histograms for all blocks
    feature_dim = (num_points + 2) * 3  # 3 planes: XY, XT, YT
    all_features = np.zeros((block_size * block_size, feature_dim))

    # Process each block
    for by in range(block_size):
        for bx in range(block_size):
            # Extract block
            y_start = by * block_height
            y_end = (by + 1) * block_height
            x_start = bx * block_width
            x_end = (bx + 1) * block_width

            block = video_frames[:, y_start:y_end, x_start:x_end]

            # Initialize histograms for current block
            hist_xy = np.zeros(num_points + 2)
            hist_xt = np.zeros(num_points + 2)
            hist_yt = np.zeros(num_points + 2)

            # XY plane (spatial texture)
            # Skip boundary frames affected by temporal radius
            for t in range(rt, num_frames - rt):
                lbp_xy = local_binary_pattern(block[t], num_points, rx, method='uniform')
                hist, _ = np.histogram(lbp_xy.ravel(),
                                       bins=num_points + 2,
                                       range=(0, num_points + 2),
                                       density=True)
                hist_xy += hist
            hist_xy = hist_xy / (num_frames - 2 * rt) if num_frames > 2 * rt else hist_xy

            # XT plane (horizontal motion)
            # Process each row over time
            for y in range(block_height):
                xt_volume = block[:, y, :]  # time-x plane for current y
                # Consider temporal radius
                for x in range(block_width):
                    lbp_xt = local_binary_pattern(xt_volume[:, x:x + 2 * rt + 1],
                                                  num_points,
                                                  rt,
                                                  method='uniform')
                    hist, _ = np.histogram(lbp_xt.ravel(),
                                           bins=num_points + 2,
                                           range=(0, num_points + 2),
                                           density=True)
                    hist_xt += hist
            hist_xt = hist_xt / (block_height * block_width)

            # YT plane (vertical motion)
            # Process each column over time
            for x in range(block_width):
                yt_volume = block[:, :, x]  # time-y plane for current x
                # Consider temporal radius
                for y in range(block_height):
                    lbp_yt = local_binary_pattern(yt_volume[:, y:y + 2 * rt + 1],
                                                  num_points,
                                                  rt,
                                                  method='uniform')
                    hist, _ = np.histogram(lbp_yt.ravel(),
                                           bins=num_points + 2,
                                           range=(0, num_points + 2),
                                           density=True)
                    hist_yt += hist
            hist_yt = hist_yt / (block_height * block_width)

            # Normalize all histograms
            hist_xy = hist_xy / np.sum(hist_xy) if np.sum(hist_xy) > 0 else hist_xy
            hist_xt = hist_xt / np.sum(hist_xt) if np.sum(hist_xt) > 0 else hist_xt
            hist_yt = hist_yt / np.sum(hist_yt) if np.sum(hist_yt) > 0 else hist_yt

            # Concatenate histograms
            block_features = np.concatenate([hist_xy, hist_xt, hist_yt])

            # Store features for current block
            all_features[by * block_size + bx] = block_features

    # Additional normalization of the final feature vector
    features_flattened = all_features.flatten()
    features_normalized = features_flattened / np.sum(features_flattened) if np.sum(
        features_flattened) > 0 else features_flattened

    return features_normalized


def process_sequence(frames):
    """
    Process a sequence of frames to extract LBP-TOP features.
    Added proper preprocessing steps.
    """
    # Convert to grayscale if needed
    if len(frames.shape) == 4:  # If RGB
        frames = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])

    # Normalize frame intensities
    frames = frames.astype(np.float32)
    frames = (frames - frames.min()) / (frames.max() - frames.min() + 1e-6)

    # Apply Gaussian smoothing to reduce noise
    frames = np.array([cv2.GaussianBlur(f, (3, 3), 0.5) for f in frames])

    # Extract features
    features = lbp_top_block(frames, rx=1, ry=1, rt=4, num_points=4, block_size=5)

    return features


# def lbp_top(video_frames, rx=1, ry=1, rt=4, num_points=8, radius=1):
#     """
#     Compute LBP-TOP features for a sequence of frames.
#     """
#     num_frames, height, width = video_frames.shape
#     lbp_xy_hist, lbp_xt_hist, lbp_yt_hist = [], [], []
#
#     for t in range(rt, num_frames - rt):
#         # XY plane
#         lbp_xy = local_binary_pattern(video_frames[t], num_points, radius, method='uniform')
#         lbp_xy_hist.append(np.histogram(lbp_xy, bins=np.arange(0, num_points + 3), density=True)[0])
#
#         # XT plane
#         lbp_xt = local_binary_pattern(video_frames[:, :, width // 2], num_points, radius, method='uniform')
#         lbp_xt_hist.append(np.histogram(lbp_xt, bins=np.arange(0, num_points + 3), density=True)[0])
#
#         # YT plane
#         lbp_yt = local_binary_pattern(video_frames[:, height // 2, :], num_points, radius, method='uniform')
#         lbp_yt_hist.append(np.histogram(lbp_yt, bins=np.arange(0, num_points + 3), density=True)[0])
#
#     # Concatenate histograms from all three planes
#     lbp_top_hist = np.concatenate([np.mean(lbp_xy_hist, axis=0),
#                                     np.mean(lbp_xt_hist, axis=0),
#                                     np.mean(lbp_yt_hist, axis=0)])
#     return lbp_top_hist

def lbp_top(video_frames, rx=1, ry=1, rt=4, num_points=8, radius=1):
    """
    Compute LBP-TOP features for a sequence of frames.
    """
    num_frames, height, width = video_frames.shape

    # Initialize histograms for all planes
    xy_hists = np.zeros((height, width, num_points + 2))
    xt_hists = np.zeros((height, num_frames, num_points + 2))
    yt_hists = np.zeros((width, num_frames, num_points + 2))

    # XY plane - compute for each frame
    for t in range(num_frames):
        xy_lbp = local_binary_pattern(video_frames[t], num_points, radius, method='uniform')
        xy_hists += np.histogramdd(xy_lbp.ravel(), bins=num_points + 2, density=True)[0]

    # XT plane - compute for each row
    for y in range(height):
        xt_lbp = local_binary_pattern(video_frames[:, y, :], num_points, radius, method='uniform')
        xt_hists[y] = np.histogram(xt_lbp, bins=num_points + 2, density=True)[0]

    # YT plane - compute for each column
    for x in range(width):
        yt_lbp = local_binary_pattern(video_frames[:, :, x], num_points, radius, method='uniform')
        yt_hists[x] = np.histogram(yt_lbp, bins=num_points + 2, density=True)[0]

    # Combine histograms
    final_hist = np.concatenate([
        xy_hists.mean(axis=(0, 1)),
        xt_hists.mean(axis=(0, 1)),
        yt_hists.mean(axis=(0, 1))
    ])

    return final_hist

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
    video_dir = r"C:\casme_crops\Cropped"  # Update with your directory path

    # Collect all episode folder paths (EPXX_XXX) inside subXX folders
    all_episode_paths = []
    for subfolder in os.listdir(video_dir):
        subfolder_path = os.path.join(video_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        for episode_folder in os.listdir(subfolder_path):
            episode_path = os.path.join(subfolder_path, episode_folder)
            if os.path.isdir(episode_path):
                all_episode_paths.append(episode_path)

    # Use multiprocessing to process episodes in parallel
    print(f"Found {len(all_episode_paths)} episode folders. Starting multiprocessing...")
    with Pool(processes=8) as pool:
        pool.map(process_episode, all_episode_paths)