from omegaconf import OmegaConf
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from einops import rearrange
from model import MotionMagModel

class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.data.append(val)
        if len(self.data) > 1:
            self.std = np.std(self.data, ddof=1)
            self.se = self.std / np.sqrt(len(self.data))
        else:
            self.std = 0  # Avoid NaN issues
            self.se = 0


    def __str__(self):
        return f"{self.name}: {self.val:.5f} {self.avg:.5f}"


def write_video(frames, fps, output_path):
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def get_our_model(args, training=False):
    # Load model configuration
    config = OmegaConf.load(args.config)
    config.config = args.config
    config.train.ngpus = 1
    config.train.is_training = training
    config.data.batch_size = 1

    print('Making model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionMagModel(config).to(device)

    # Load checkpoint
    print(f'Resuming from {args.resume}')
    chkpt = torch.load(args.resume, map_location=device)

    # Extract state_dict if present
    state_dict = chkpt.get("state_dict", chkpt)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}  # Remove 'module.' prefix if needed

    # Try loading the state_dict
    try:
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ Model loaded successfully.")
    except RuntimeError as e:
        print(f"❌ Error loading state_dict: {e}")
        raise ValueError("🚨 Model checkpoint does not match the model architecture.")

    # Wrap model in DataParallel only if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("🖥️ Using DataParallel for multiple GPUs.")

    # Return model and stored epoch number from checkpoint
    return model, chkpt.get("epoch", 0)


def dist_transform(mask):
    h, w = mask.shape

    def closest_se(mask):
        closest = torch.full(mask.shape, float('inf'))
        for i in range(1, h):
            for j in range(1, w):
                if mask[i, j] == 1:
                    closest[i, j] = 0
                else:
                    closest[i, j] = min(closest[i, j - 1], closest[i - 1, j]) + 1
        return closest

    se = closest_se(mask)
    nw = closest_se(mask.flip((0, 1))).flip((0, 1))
    sw = closest_se(mask.flip(0)).flip(0)
    ne = closest_se(mask.flip(1)).flip(1)

    return torch.minimum(torch.minimum(se, nw), torch.minimum(sw, ne))


def log_images(writer, images_dict, epoch_num, split, config):
    save_dir = config.save_dir / 'images'
    save_dir.mkdir(exist_ok=True, parents=True)

    for name, images in images_dict.items():
        for idx, image in enumerate(rearrange(images, 'b c t h w -> t b c h w')):
            grid = make_grid(image)
            writer.add_image(f'{name}_{idx}/{split}', grid, epoch_num)
            save_image(grid, save_dir / f'epoch{epoch_num:05}.{split}.{name}{idx:02}.png')
