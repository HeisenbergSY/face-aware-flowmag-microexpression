
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

def generate_binary_mask(image, img_size=512):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_points = []
            for lm in face_landmarks.landmark:
                x = int(lm.x * img_size)
                y = int(lm.y * img_size)
                landmark_points.append((x, y))
            # Fill convex hull of all landmarks
            landmark_points = np.array(landmark_points, dtype=np.int32)
            cv2.fillConvexPoly(mask, landmark_points, 255)

    return mask

def generate_static_mask_overlay_montage(
    image_dir,
    output_path='binary_mask_overlay_montage.png',
    num_frames=5
):
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    if len(image_paths) == 0:
        raise ValueError("No .jpg files found in image_dir")

    step = max(1, len(image_paths) // num_frames)
    selected_indices = list(range(0, len(image_paths), step))[:num_frames]

    fig, axs = plt.subplots(2, num_frames, figsize=(3*num_frames, 6))

    for i, idx in enumerate(selected_indices):
        img = cv2.imread(image_paths[idx])
        img = cv2.resize(img, (512, 512))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Generate binary mask from landmarks
        mask = generate_binary_mask(img)

        # Overlay mask
        overlay = rgb.copy()
        overlay[mask > 0] = [255, 0, 0]
        overlay = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)

        axs[0, i].imshow(rgb)
        axs[0, i].set_title(f'Frame {idx}')
        axs[0, i].axis('off')

        axs[1, i].imshow(overlay)
        axs[1, i].set_title(f'Overlay {idx}')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Binary mask montage saved to {output_path}")
    plt.show()

generate_static_mask_overlay_montage(
    image_dir=r"C:\Users\thepr\Desktop\Master\flowmag\Cropped\Cropped\sub02\EP02_04f"
)
