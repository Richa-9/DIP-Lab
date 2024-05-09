import numpy as np
import cv2
import matplotlib.pyplot as plt

def harmonic_filter(image):
    filter_kernel = np.ones((3, 3), dtype=np.float32) / (3 ** 2)

    filtered_image = cv2.filter2D(image, -1, filter_kernel)
    return filtered_image

image_path = "Lenna.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

filtered_image = harmonic_filter(image)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Harmonic Filtered Image')
plt.show()
