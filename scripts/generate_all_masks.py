from pathlib import Path
from generate_face_masks import generate_mask

root = Path("Cropped")  # Adjust if needed

for subject in root.iterdir():
    for sequence in subject.iterdir():
        if not sequence.is_dir():
            continue

        frames = sorted(sequence.glob("*.jpg"))
        if not frames:
            continue

        first_frame = frames[0]
        output_path = sequence / "face_mask.npy"

        generate_mask(first_frame, output_path)
