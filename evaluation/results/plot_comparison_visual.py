
import os
import cv2
import matplotlib.pyplot as plt

def load_images_from_folder(folder, max_images=3):
    images = []
    filenames = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])
    for filename in filenames[:max_images]:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images

def plot_comparison(original_dir, mag1_dir, mag2_dir, labels, output_path='visual_comparison.png'):
    orig_imgs = load_images_from_folder(original_dir)
    mag1_imgs = load_images_from_folder(mag1_dir)
    mag2_imgs = load_images_from_folder(mag2_dir)

    n = min(len(orig_imgs), len(mag1_imgs), len(mag2_imgs))
    fig, axs = plt.subplots(n, 3, figsize=(12, 4 * n))

    for i in range(n):
        axs[i, 0].imshow(orig_imgs[i])
        axs[i, 0].set_title('Original')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(mag1_imgs[i])
        axs[i, 1].set_title(labels[0])
        axs[i, 1].axis('off')

        axs[i, 2].imshow(mag2_imgs[i])
        axs[i, 2].set_title(labels[1])
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved comparison figure to: {output_path}")
    plt.show()

# Example usage:
# plot_comparison(
#     original_dir='path/to/original',
#     mag1_dir='path/to/mag_lambda2',
#     mag2_dir='path/to/mag_lambda5',
#     labels=['Magnified (λ=2)', 'Magnified (λ=5)']
# )
