from pathlib import Path
from draw_landmarks_triplet import draw_landmarks_triplet

image_path = Path(r"C:\Users\thepr\Desktop\Master\flowmag\Cropped\Cropped\sub01\EP02_01f/reg_img46.jpg")
output_path = Path("output/landmarks_triplet.jpg")

draw_landmarks_triplet(image_path, output_path, img_size=512)
