import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def visualize_landmark_mask(image_path, mask_path, output_path='landmark_mask_visualization.png'):
    # Load the original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load the binary mask
    mask = np.load(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # 🔧 Resize mask to match image size
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Normalize and scale mask for visualization
    mask_vis = (mask * 255).astype(np.uint8)

    # Create overlay visualization
    overlay = img.copy()
    overlay[mask > 0.1] = [255, 0, 0]  # Red overlay where mask is active
    overlay = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    # Plot all
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img)
    axs[0].set_title('Original Frame')
    axs[0].axis('off')

    axs[1].imshow(mask_vis, cmap='gray')
    axs[1].set_title('Binary Landmark Mask')
    axs[1].axis('off')

    axs[2].imshow(overlay)
    axs[2].set_title('Overlay')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved visualization to {output_path}")
    plt.show()


# ✅ Put the call OUTSIDE the function
visualize_landmark_mask(
    image_path=r"C:\Users\thepr\Desktop\Master\flowmag\Cropped\Cropped\sub02\EP02_04f\reg_img31.jpg",
    mask_path=r"C:\Users\thepr\Desktop\Master\flowmag\Cropped\Cropped\sub02\EP02_04f\face_mask.npy"
)
