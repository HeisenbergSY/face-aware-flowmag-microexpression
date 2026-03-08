# draw_landmarks_triplet.py
import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

def draw_landmarks_triplet(image_path, output_path, img_size=512):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to read {image_path}")
        return

    image = cv2.resize(image, (img_size, img_size))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    blank = np.zeros_like(image)
    overlay = image.copy()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                x = int(lm.x * img_size)
                y = int(lm.y * img_size)
                cv2.circle(blank, (x, y), 2, (0, 255, 0), -1)
                cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)

        combined = np.concatenate((image, blank, overlay), axis=1)
        os.makedirs(output_path.parent, exist_ok=True)
        cv2.imwrite(str(output_path), combined)
        print(f"✅ Saved triplet image to {output_path}")
    else:
        print(f"⚠️ No face detected in {image_path}")
