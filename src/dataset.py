import os
import json
from pathlib import Path
import random
import torch.nn.functional as F
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation, resize
from torchvision.transforms import RandomResizedCrop, Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, CenterCrop, Resize, ColorJitter, ToTensor

from einops import rearrange

class RepeatDataset(Dataset):
    def __init__(self, dataset, factor=10):
        self.dataset = dataset
        self.factor = factor

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        return self.dataset[idx]

    def __len__(self):
        return self.factor * len(self.dataset)


class TrainingFramesDataset(Dataset):
    def __init__(self, root, im_size=256):
        self.root = Path(root)
        self.im_size = im_size
        self.sequence_dirs = sorted([p for p in self.root.glob("sub*/EP*/") if p.is_dir()])
        self.frame_paths = []

        for seq_dir in self.sequence_dirs:
            self.frame_paths.extend(sorted(seq_dir.glob("*.jpg")))
            self.frame_paths.extend(sorted(seq_dir.glob("*.png")))

        print(f"✅ Found {len(self.frame_paths)} total frames in {len(self.sequence_dirs)} sequences.")

    def transform_frame(self, frame):
        frame = resize(frame, (self.im_size, self.im_size))
        c, h, w = frame.shape
        frame = frame[:, :(h // 8) * 8, :(w // 8) * 8]
        return frame

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        frame = Image.open(frame_path).convert("RGB")
        frame = to_tensor(frame)
        frame = self.transform_frame(frame)

        # Load mask from the same folder
        mask_path = frame_path.parent / "face_mask.npy"
        if mask_path.exists():
            mask = np.load(mask_path)
            mask = torch.from_numpy(mask).float()
            # print(f"Loaded mask: {mask.shape}, min: {mask.min():.3f}, max: {mask.max():.3f}")
            # Resize mask to match image size
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),  # (1,1,H,W)
                size=(self.im_size, self.im_size),
                mode='nearest'
            ).squeeze(0).squeeze(0)  # (H, W)
        else:
            _, h, w = frame.shape
            mask = torch.ones((h, w)).float()


        frames = torch.stack([frame, frame], dim=1)

        return frames, {"mask": mask}

    def __len__(self):
        return len(self.frame_paths)


class FramesDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        # Filter only image files
        self.frame_names = sorted([
            fname for fname in os.listdir(self.root)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])


    def transform_frame(self, frame):
        c, h, w = frame.shape
        frame = frame[:,:(h//8)*8,:(w//8)*8]
        return frame

    def __getitem__(self, idx):
        fname = self.frame_names[idx]
        frame = Image.open(self.root / fname)
        frame = to_tensor(frame)
        return frame

    def __len__(self):
        return len(self.frame_names)


class TestTimeAdaptDataset(Dataset):
    def __init__(self, root, mode='first', length=None):
        self.root = Path(root)
        self.frame_names = sorted(os.listdir(self.root))
        self.mode = mode
        self.im_size = 256
        self.scale = 1.1

        self.geom_transform = Compose([
            RandomRotation(5),
        ])
        self.color_transform = ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.3)

        self.length = length if length else len(self.frame_names)

        self.cache = []
        to_tensor = ToTensor()
        print(f"📥 Loading {len(self.frame_names)} frames into RAM...")

        for fname in self.frame_names:
            img = Image.open(self.root / fname).convert("RGB")
            self.cache.append(to_tensor(img))

        print(f"✅ Finished loading {len(self.cache)} frames into RAM!")

    def __getitem__(self, idx):
        return self.cache[idx]

    def __len__(self):
        return self.length

    def transform_frames(self, frames):
        c, t, h, w = frames.shape
        frames = rearrange(frames, 'c t h w -> (c t) h w')
        frames = self.geom_transform(frames)
        frames = rearrange(frames, '(c t) h w -> c t h w', c=c, t=t)

        new_h = int(h * self.scale**np.random.uniform(-1, 1))
        new_w = int(w * self.scale**np.random.uniform(-1, 1))
        frames = resize(frames, (new_h, new_w))

        to_pad_h = max(self.im_size - new_h, 0)
        to_pad_w = max(self.im_size - new_w, 0)
        pad_l_h = to_pad_h // 2
        pad_r_h = to_pad_h - pad_l_h
        pad_l_w = to_pad_w // 2
        pad_r_w = to_pad_w - pad_l_w
        frames = F.pad(frames, (pad_l_w, pad_r_w, pad_l_h, pad_r_h))
        _, _, padded_h, padded_w = frames.shape

        ch = self.im_size
        cw = self.im_size
        ct = np.random.randint(padded_h-ch+1)
        cl = np.random.randint(padded_w-cw+1)
        frames = frames[:,:,ct:ct+ch,cl:cl+cw]

        frames = rearrange(frames, 'c t h w -> t c h w')
        frames = self.color_transform(frames)
        frames = rearrange(frames, 't c h w -> c t h w')

        return frames

    def __getitem__(self, idx):
        fname = self.frame_names[idx]
        frame_path = self.root / fname

        frame = Image.open(frame_path)
        frame = to_tensor(frame)
        frame = self.transform_frame(frame)

        sequence_dir = frame_path.parent
        mask_path = sequence_dir / "face_mask.npy"

        if mask_path.exists():
            mask = np.load(mask_path)
            mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, size=(self.im_size, self.im_size), mode='nearest')
            mask = mask.squeeze(0)
        else:
            _, h, w = frame.shape
            mask = torch.ones((1, h, w)).float()

        return frame, {'mask': mask}


class FlowMagDataset(Dataset):
    def __init__(self, data_root, split, aug=False, img_size=256):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size

        if self.split == 'valid':
            self.split = 'test'

        self.frameA_dir = self.data_root / self.split / 'frameA'
        self.frameB_dir = self.data_root / self.split / 'frameB'
        with open(self.data_root / f'{self.split}_fn.json', 'r') as f:
            self.fnames = json.load(f)

        if aug:
            self.transform = Compose([
                RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                RandomHorizontalFlip(.5),
                RandomVerticalFlip(.5),
                RandomRotation(15),
            ])
            self.color_transform = ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.3)
        else:
            self.transform = Compose([
                Resize(img_size),
                CenterCrop(img_size),
            ])
            self.color_transform = nn.Identity()

    def transform_frames(self, frames):
        c, t, h, w = frames.shape
        frames = rearrange(frames, 'c t h w -> (c t) h w')
        frames = self.transform(frames)
        frames = rearrange(frames, '(c t) h w -> c t h w', c=c, t=t)
        frames = rearrange(frames, 'c t h w -> t c h w')
        frames = self.color_transform(frames)
        frames = rearrange(frames, 't c h w -> c t h w')
        return frames

    def __getitem__(self, idx):
        image_paths = [self.frameA_dir / self.fnames[idx], self.frameB_dir / self.fnames[idx]]
        images = [Image.open(path) for path in image_paths]
        images = [to_tensor(im) for im in images]
        frames = torch.stack(images, dim=1)

        frames = self.transform_frames(frames)

        info = {'fname': self.fnames[idx]}
        return frames, info

    def __len__(self):
        return len(self.fnames)

def get_dataloader(config, split):
    if split == 'train':
        aug = config.data.aug
    else:
        aug = False

    dataset = FlowMagDataset(config.data.dataroot, split=split, aug=aug, img_size=config.data.im_size)
    return dataset
