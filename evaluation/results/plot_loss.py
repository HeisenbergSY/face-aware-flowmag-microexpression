import re
import matplotlib.pyplot as plt
import os

def plot_loss_curves(log_file, output_name='loss_curves_first_20_epochs.svg', max_epoch=20):
    # Read log lines
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Extract directory from log path
    output_dir = os.path.dirname(log_file)
    output_path = os.path.join(output_dir, output_name)

    # Define regex pattern for loss values
    pattern = re.compile(
        r"\[Epoch (\d+)\] \[Iter \d+/\d+\] Loss - ([\d.]+) \| MagLoss - ([\d.]+) \| ColorLoss - ([\d.]+) \| LandmarkLoss - ([\d.]+)"
    )

    iterations = []
    losses = {'Loss': [], 'MagLoss': [], 'ColorLoss': [], 'LandmarkLoss': []}

    # Parse the log file
    for line in lines:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                break
            iterations.append(len(iterations))
            losses['Loss'].append(float(match.group(2)))
            losses['MagLoss'].append(float(match.group(3)))
            losses['ColorLoss'].append(float(match.group(4)))
            losses['LandmarkLoss'].append(float(match.group(5)))

    # Plotting
    plt.figure(figsize=(12, 8))
    for key, values in losses.items():
        plt.plot(iterations, values, label=key)
    
    plt.title('Training Loss Curves (First 20 Epochs)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format='svg')
    print(f"✅ Saved plot to: {output_path}")

# Example usage
plot_loss_curves(
    r"C:\Users\thepr\Desktop\Master\flowmag\results\03_26_2025-09-14-49-alpha16.color10.raft\logs.txt"
)
