import os
import re
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_training_validation_loss_smooth(log_file, window_size=3):
    log_dir = os.path.dirname(log_file)
    output_path = os.path.join(log_dir, 'training_vs_validation_loss_smoothed.svg')

    train_pattern = re.compile(r"\[Epoch (\d+)\].*?Loss - ([\d.]+)")
    val_pattern = re.compile(r"\[Epoch (\d+)\] \[Validation\] Loss - ([\d.]+)")

    train_losses = {}
    val_losses = {}

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            train_match = train_pattern.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                loss = float(train_match.group(2))
                train_losses.setdefault(epoch, []).append(loss)
            val_match = val_pattern.search(line)
            if val_match:
                val_losses[int(val_match.group(1))] = float(val_match.group(2))

    epochs = sorted(set(train_losses) | set(val_losses))
    avg_train_losses = [np.mean(train_losses[e]) for e in epochs if e in train_losses]
    val_loss_list = [val_losses.get(e, np.nan) for e in epochs]

    # Apply smoothing
    smooth_train = moving_average(avg_train_losses, window_size)
    smooth_val = moving_average(val_loss_list, window_size)

    # Adjust epochs due to moving average shrinking
    smooth_epochs = epochs[window_size - 1:]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(smooth_epochs, smooth_train, label=f'Training Loss (MA {window_size})', marker='o')
    plt.plot(smooth_epochs, smooth_val, label=f'Validation Loss (MA {window_size})', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Smoothed Training vs Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format='svg')
    plt.close()
    print(f"✅ Smoothed plot saved to: {output_path}")

# Example usage
plot_training_validation_loss_smooth(r"C:\Users\thepr\Desktop\Master\flowmag\results\03_25_2025-20-13-34-alpha16.color10.raft\logs.txt", window_size=7)
