import os
import subprocess
from pathlib import Path

# === YOUR SETTINGS ===
root_dir = Path(r"C:\Users\thepr\Desktop\Master\flowmag\Cropped\Cropped")
resume = "checkpoints/raft_chkpt_00140.pth"
config = "configs/alpha16.color10.yaml"
script_path = r"C:\Users\thepr\Desktop\Master\flowmag\inference.py"
alpha = 20
output_video = True

# ======================

# Walk through all subdirectories (subXX/EP folders)
# Only keep folders that have at least one .jpg frame
all_frame_dirs = [
    p for p in root_dir.rglob("*") 
    if p.is_dir() and any(f.suffix == ".jpg" for f in p.iterdir())
]


for frames_dir in sorted(all_frame_dirs):
    save_name = frames_dir.parts[-2] + "_" + frames_dir.name  # e.g., sub01_EP02_01f

    print(f"\n>>> Running inference on {save_name} <<<\n")

    cmd = [
        "python", script_path,
        "--config", config,
        "--frames_dir", str(frames_dir),
        "--resume", resume,
        "--save_name", save_name,
        "--alpha", str(alpha),
    ]

    if output_video:
        cmd.append("--output_video")
    mask_path = frames_dir / "face_mask.npy"
    if mask_path.exists():
        cmd += ["--mask_path", str(mask_path)]
    else:
        print(f"⚠️ No mask found in {frames_dir}, running without a mask.")

    subprocess.run(cmd)
