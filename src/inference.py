# For Landmark magnification
import argparse
from tqdm import tqdm

from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.utils import save_image

from src.dataset import TrainingFramesDataset, FramesDataset
from src.test_time_adapt import test_time_adapt
from src.myutils import get_our_model, write_video, dist_transform

def inference(model, 
              frames_dataset, 
              save_dir, 
              alpha=2.0, 
              max_alpha=16.0, 
              mask=None, 
              num_device=1, 
              output_video=False):

    device = 'cuda'
    save_dir.mkdir(exist_ok=True, parents=True)
    results = []

    if isinstance(model, nn.DataParallel):
        model.module.eval()
    else:
        model.eval()

    # Infer whether model supports training-mode forward pass
    supports_training = hasattr(model, 'module') and hasattr(model.module, 'get_training_status')
    if supports_training:
        use_training_forward = model.module.get_training_status()
    else:
        use_training_forward = True

    if alpha > max_alpha and np.sqrt(alpha) < max_alpha:
        our_alpha = np.sqrt(alpha)
        num_recursion = 2
    elif alpha <= max_alpha:
        our_alpha = alpha
        num_recursion = 1
    else:
        raise Exception('alpha out of range')

    with torch.no_grad():
        im0 = frames_dataset[0][None].to(device)
        results.append(im0.detach().cpu())

        for i in tqdm(range(1, len(frames_dataset))):
            im1 = frames_dataset[i][None].to(device)
            frames = torch.stack([im0, im1], dim=2).repeat(num_device,1,1,1,1)

            for _ in range(num_recursion):
                if use_training_forward:
                    pred, _, _ = model(frames, alpha=our_alpha, mask=mask)
                else:
                    pred = model(frames, alpha=our_alpha, mask=mask)
                frames = torch.stack([im0, pred[0,:,0].unsqueeze(0)], dim=2).repeat(num_device,1,1,1,1)

            pred = pred[0,:,0]
            results.append(pred.detach().cpu())

    if output_video:
        saved_frames = [(255*img.squeeze().permute(1,2,0).flip([-1]).numpy()).astype(np.uint8) for img in results]
        video_path = str(save_dir / f'x{alpha}.mp4') if mask is None else str(save_dir / f'masked_x{alpha}.mp4')
        write_video(saved_frames, 30, video_path)
        print('saved the video to {}'.format(video_path))
    else:
        save_dir = save_dir / f'x{alpha}' if mask is None else save_dir / f'masked_x{alpha}'
        save_dir.mkdir(exist_ok=True, parents=True)
        for i, img in enumerate(results):
            save_image(img, save_dir / f'{i+1:04}.png')
        print('saved the images to {}'.format(str(save_dir)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--frames_dir', type=str, required=True)
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--save_name', type=str, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--mask_path', type=str, default=None)
    parser.add_argument('--soft_mask', type=int, default=0)
    parser.add_argument('--output_video', action='store_true')
    parser.add_argument('--test_time_adapt', action='store_true')
    parser.add_argument('--tta_epoch', type=int, default=3)
    args = parser.parse_args()

    frames_dataset = TrainingFramesDataset(args.frames_dir) if args.test_time_adapt else FramesDataset(args.frames_dir)

    mask = None
    if args.mask_path:
        mask = np.load(args.mask_path)
        mask = torch.tensor(mask).float()

        # Clean up to always get [B=1, C=1, H, W]
        # Ensure it's exactly [1, H, W]
        while mask.ndim > 3:
            mask = mask.squeeze(0)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)

        # Final shape: [1, 1, H, W]
        print(f"✅ Loaded mask with shape: {mask.shape}")
        if args.soft_mask:
            print('Softening mask')
            dist = dist_transform(mask)
            dist[dist < args.soft_mask] = 1
            dist[dist >= args.soft_mask] = 0
            mask = dist

    config = OmegaConf.load(args.config)
    max_alpha = config.train.alpha_high

    save_dir = Path(args.resume).parent.parent / 'inference' / args.save_name
    save_dir.mkdir(exist_ok=True, parents=True)

    model, epoch = get_our_model(args, args.test_time_adapt)

    if args.test_time_adapt:
        save_dir = save_dir / f'tta_epoch{epoch:03}'
        save_dir.mkdir(exist_ok=True, parents=True)

        def inference_fn(model, epoch):
            new_save_dir = save_dir / f'tta_epoch{epoch:03}'
            new_save_dir.mkdir(exist_ok=True, parents=True)
            inference(model, frames_dataset, new_save_dir, alpha=args.alpha, max_alpha=max_alpha, mask=mask, num_device=1, output_video=args.output_video)

        model, loss_info = test_time_adapt(model, args.frames_dir, num_epochs=args.tta_epoch, inference_fn=inference_fn, inference_freq=1, alpha=None, save_dir=save_dir, dataset_length=1000)

        for loss_name, losses in loss_info.items():
            plt.plot(losses)
            plt.title(loss_name)
            plt.savefig(save_dir / f'{loss_name}.png')
            plt.clf()

    inference(model, frames_dataset, save_dir, alpha=args.alpha, max_alpha=max_alpha, mask=mask, num_device=1, output_video=args.output_video)
