import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

frame = Image.open("Cropped\Cropped\sub01\EP02_01f\\reg_img46.jpg")
mask = np.load("Cropped/Cropped/sub01/EP02_01f/face_mask.npy")

print(mask.shape, mask.min(), mask.max(), mask.mean())
plt.subplot(1, 2, 1)
plt.imshow(frame)
plt.title("Original Frame")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title("Face Mask")
plt.show()
