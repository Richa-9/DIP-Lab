import cv2
import numpy as np
import matplotlib.pyplot as plt

def impulse_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)

    salt_pixels = np.random.rand(*image.shape) < salt_prob
    noisy_image[salt_pixels] = 255

    pepper_pixels = np.random.rand(*image.shape) < pepper_prob
    noisy_image[pepper_pixels] = 0

    return noisy_image
def median_filter(image, kernel_size):

    filtered_image = np.zeros_like(image)
    pad_size = kernel_size // 2

    padded_image = np.pad(image, pad_size, mode='constant')

    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            patch = padded_image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            median = np.median(patch, axis=(0, 1))
            filtered_image[i-pad_size, j-pad_size] = median

    return filtered_image

image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

noisy_img = impulse_noise(image, salt_prob=0.05, pepper_prob=0.05)

restored_img = median_filter(noisy_img, kernel_size=3)

plt.figure(figsize=(10, 8))

plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2), plt.imshow(noisy_img, cmap='gray')
plt.title('Noisy Image')

plt.subplot(1, 3, 3), plt.imshow(restored_img, cmap='gray')
plt.title('restored Image')

plt.tight_layout()
plt.show()
