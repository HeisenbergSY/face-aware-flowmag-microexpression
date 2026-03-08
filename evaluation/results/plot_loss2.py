import os
import re
import matplotlib.pyplot as plt

def plot_training_validation_loss(log_file):
    # Prepare paths
    log_dir = os.path.dirname(log_file)
    output_path = os.path.join(log_dir, 'training_vs_validation_loss.svg')

    # Patterns
    train_pattern = re.compile(r"\[Epoch (\d+)\].*?Loss - ([\d.]+)")
    val_pattern = re.compile(r"\[Epoch (\d+)\] \[Validation\] Loss - ([\d.]+)")

    # Containers
    train_losses = {}
    val_losses = {}

    # Parse log
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            train_match = train_pattern.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                loss = float(train_match.group(2))
                if epoch not in train_losses:
                    train_losses[epoch] = []
                train_losses[epoch].append(loss)
            val_match = val_pattern.search(line)
            if val_match:
                val_losses[int(val_match.group(1))] = float(val_match.group(2))

    # Average training loss per epoch
    epochs = sorted(set(train_losses.keys()).union(val_losses.keys()))
    avg_train_losses = [sum(train_losses[e])/len(train_losses[e]) for e in epochs if e in train_losses]
    val_loss_list = [val_losses.get(e, None) for e in epochs]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs[:len(avg_train_losses)], avg_train_losses, label='Training Loss', marker='o')
    plt.plot(epochs[:len(val_loss_list)], val_loss_list, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, format='svg')
    plt.close()
    print(f"✅ Plot saved to: {output_path}")

# Example usage
plot_training_validation_loss(r"C:\Users\thepr\Desktop\Master\flowmag\results\03_26_2025-09-14-49-alpha16.color10.raft\logs.txt")
