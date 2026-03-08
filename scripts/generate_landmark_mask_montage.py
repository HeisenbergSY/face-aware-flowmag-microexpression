# This script generates a montage of images and their corresponding masks.
# It takes a directory of images and a directory of masks, and creates a montage
# of the images and their corresponding masks side by side.
# The montage is saved as a PNG file.
# The script uses OpenCV for image processing and Matplotlib for visualization.

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_static_mask_overlay_montage(
    image_dir,
    mask_path,
    output_path='landmark_mask_overlay_montage.png',
    num_frames=5
):
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    if len(image_paths) == 0:
        raise ValueError("No .jpg files found in image_dir")

    # Evenly space frames
    step = max(1, len(image_paths) // num_frames)
    selected_indices = list(range(0, len(image_paths), step))[:num_frames]

    # Load mask
    mask = np.load(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    fig, axs = plt.subplots(2, num_frames, figsize=(3*num_frames, 6))

    for i, idx in enumerate(selected_indices):
        img = cv2.imread(image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize mask to match image size
        resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Overlay mask
        overlay = img.copy()
        overlay[resized_mask > 0.1] = [255, 0, 0]
        overlay = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        axs[0, i].imshow(img)
        axs[0, i].set_title(f'Frame {idx}')
        axs[0, i].axis('off')

        axs[1, i].imshow(overlay)
        axs[1, i].set_title(f'Overlay {idx}')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Overlay montage saved to {output_path}")
    plt.show()

generate_static_mask_overlay_montage(
    image_dir=r"C:\Users\thepr\Desktop\Master\flowmag\Cropped\Cropped\sub02\EP02_04f",
    mask_path=r"C:\Users\thepr\Desktop\Master\flowmag\Cropped\Cropped\sub02\EP02_04f\face_mask.npy"
)
