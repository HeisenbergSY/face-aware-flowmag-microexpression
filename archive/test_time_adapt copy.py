from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from src.dataset import TestTimeAdaptDataset
from src.myutils import AverageMeter
import matplotlib.pyplot as plt

def test_time_adapt(model, frames_dir, num_epochs=10, mode='first', device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), inference_fn=None, inference_freq=1, alpha=None, save_dir=None, dataset_length=None):
    '''
    params:
        model: (nn.Module) model with checkpoint already loaded
        frames_dir: (string) path to directory of frames for test time adaptation
        num_epochs: (int) number of passes through the frames_dir
        mode: ['first', 'random'] how to sample frames
        device: device to put model and data on
        inference_fn: function to call at the end of each epoch

    output:
        model: (nn.Module) finetuned input module
    '''

    model.train()
    model = model.to(device)
    dataset = TestTimeAdaptDataset(frames_dir, mode=mode, length=dataset_length)
    import multiprocessing
    num_workers = min(8, multiprocessing.cpu_count() // 2)  # Uses half of CPU cores
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=num_workers, drop_last=False, pin_memory=True)


    optimizer = Adam(model.parameters(), lr=1e-4)  # Removed .module to avoid DataParallel issues
    
    # Early Stopping Parameters
    patience = 3  # Stop if no improvement after 3 epochs
    delta = 0.001  # Minimum improvement threshold
    best_loss = float("inf")
    epochs_without_improvement = 0

    meter_loss = AverageMeter('loss')
    meter_mag_loss = AverageMeter('loss_mag')
    meter_color_loss = AverageMeter('loss_color')
    hist_loss = []
    hist_mag_loss = []
    hist_color_loss = []

    for epoch in range(num_epochs):
        print(f'📌 Epoch {epoch+1}/{num_epochs}')

        if inference_fn is not None and epoch % inference_freq == 0:
            print('Performing inference...')
            model.eval()
            inference_fn(model, epoch)
            model.train()

        if save_dir is not None:
            print('Saving epoch checkpoint...')
            chkpt = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(chkpt, save_dir / f'chkpt_{epoch:04}.pth')

        total_loss = 0

        for cur_iter, frames in enumerate(tqdm(dataloader)):
            frames = frames.to(device)
            preds, loss, info = model(frames, alpha=alpha)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            meter_loss.update(loss.item())
            meter_mag_loss.update(info['loss_mag'].item())
            meter_color_loss.update(info['loss_color'].item())

        avg_loss = total_loss / len(dataloader)
        print(f'📉 Avg Loss: {avg_loss:.6f}')

        if avg_loss < best_loss - delta:
            best_loss = avg_loss
            epochs_without_improvement = 0
            print("✅ Best Loss Improved! Saving Model...")
            torch.save(model.state_dict(), save_dir / "best_tta_model.pth")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No Significant Improvement for {epochs_without_improvement}/{patience} epochs.")

        if epochs_without_improvement >= patience:
            print("⏹️ Early Stopping Triggered! Stopping TTA.")
            break

        hist_loss.append(meter_loss.avg)
        hist_mag_loss.append(meter_mag_loss.avg)
        hist_color_loss.append(meter_color_loss.avg)
        meter_loss.reset()
        meter_mag_loss.reset()
        meter_color_loss.reset()

        if save_dir is not None:
            loss_info = {'losses': hist_loss, 'mag_losses': hist_mag_loss, 'color_losses': hist_color_loss}
            for loss_name, losses in loss_info.items():
                if len(losses) > 0:
                    plt.plot(losses)
                    plt.title(loss_name)
                    plt.savefig(save_dir / f'{loss_name}.png')
                    plt.clf()

    model.eval()
    if inference_fn is not None:
        inference_fn(model, epoch)

    info = {'losses': hist_loss, 'mag_losses': hist_mag_loss, 'color_losses': hist_color_loss}

    return model, info
