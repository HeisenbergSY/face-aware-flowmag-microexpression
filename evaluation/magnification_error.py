import sys
import os
sys.path.append(os.path.abspath('./flow_models'))
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import cv2  # Make sure this is imported at the top
sys.path.append(r'C:\Users\thepr\Desktop\Master\flowmag\flow_models')
from flow_models.raft.raft import RAFT
from flow_models.raft.raft_utils.utils import InputPadder
from argparse import Namespace
import glob

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load RAFT model
def load_raft_model():
    model = RAFT(Namespace(small=False, mixed_precision=False, alternate_corr=False))
    
    checkpoint = torch.load('models/raft-things.pth', map_location=DEVICE)
    state_dict = checkpoint
    
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# Compute optical flow using RAFT
def compute_flow(model, image1, image2):
    transform = transforms.Compose([transforms.ToTensor()])
    image1 = transform(image1).unsqueeze(0).to(DEVICE)
    image2 = transform(image2).unsqueeze(0).to(DEVICE)

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    with torch.no_grad():
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    return flow_up[0].permute(1, 2, 0).cpu().numpy()

# Compute average motion error
def compute_motion_error(flow1, flow2, alpha):
    # Resize flow2 to match flow1
    if flow1.shape != flow2.shape:
        flow2 = cv2.resize(flow2, (flow1.shape[1], flow1.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return np.mean(np.linalg.norm(flow2 - alpha * flow1, axis=2))


# Main evaluation function
def evaluate_motion_error(original_root, magnified_root, alpha=2.0):
    raft_model = load_raft_model()
    total_error = []
    count = 0

    subjects = sorted(os.listdir(original_root))
    for subj in subjects:
        orig_subj_path = os.path.join(original_root, subj)
        mag_subj_path = os.path.join(magnified_root, subj)
        if not os.path.isdir(orig_subj_path):
            continue

        sequences = sorted(os.listdir(orig_subj_path))
        for seq in sequences:
            orig_seq_path = os.path.join(orig_subj_path, seq)
            mag_seq_path = os.path.join(mag_subj_path, seq)
            orig_frames = sorted(glob.glob(os.path.join(orig_seq_path, "*.jpg")))
            mag_frames = sorted(glob.glob(os.path.join(mag_seq_path, "*.jpg")))

            n = min(len(orig_frames), len(mag_frames)) - 1
            for i in range(n):
                img1_orig = Image.open(orig_frames[i]).convert("RGB")
                img2_orig = Image.open(orig_frames[i+1]).convert("RGB")
                img1_mag = Image.open(mag_frames[i]).convert("RGB")
                img2_mag = Image.open(mag_frames[i+1]).convert("RGB")

                flow_orig = compute_flow(raft_model, img1_orig, img2_orig)
                flow_mag = compute_flow(raft_model, img1_mag, img2_mag)

                error = compute_motion_error(flow_orig, flow_mag, alpha)
                total_error.append(error)
                count += 1

    avg_error = np.mean(total_error)
    print(f"Average motion error: {avg_error:.6f} over {count} frame pairs")

# Example usage:
evaluate_motion_error(r'C:\Users\thepr\Desktop\Master\CASME II\Cropped', r'C:\Users\thepr\Desktop\Master\flowmag\inference\output_casme2_cropped', alpha=16.0)
