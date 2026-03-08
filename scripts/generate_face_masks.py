import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

def generate_mask(image_path, output_path, img_size=512):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to read {image_path}")
        return

    image = cv2.resize(image, (img_size, img_size))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                x = int(lm.x * img_size)
                y = int(lm.y * img_size)
                cv2.circle(mask, (x, y), radius=2, color=255, thickness=-1)

        # Dilate the points to make it more like a face-region mask
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Save the mask as a normalized numpy array
        os.makedirs(output_path.parent, exist_ok=True)
        np.save(output_path, mask.astype(np.float32) / 255.0)
        print(f"✅ Saved mask to {output_path}")
    else:
        print(f"⚠️ No face detected in {image_path}")
