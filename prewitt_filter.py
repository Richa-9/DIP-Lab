import cv2
import numpy as np
import matplotlib.pyplot as plt

def pad_image(image, pad_size):

    padded_image = np.pad(image, pad_size, mode='constant')
    return padded_image

def prewitt_filter(image):
    padded_image = pad_image(image, 1)

    x = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    gx_image = np.zeros_like(image)
    gy_image = np.zeros_like(image)
    output_image = np.zeros_like(image)

    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            gx = np.sum(x * padded_image[i - 1:i + 2, j - 1:j + 2])
            gy = np.sum(y * padded_image[i - 1:i + 2, j - 1:j + 2])
            gx_image[i - 1, j - 1] = gx
            gy_image[i - 1, j - 1] = gy
            output_image[i - 1, j - 1] = np.sqrt(gx ** 2 + gy ** 2)

    return gx_image, gy_image, output_image

image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

gx_image, gy_image, filtered_image = prewitt_filter(image)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2), plt.imshow(gx_image, cmap='gray')
plt.title('gx')


plt.subplot(2, 2, 3), plt.imshow(gy_image, cmap='gray')
plt.title('gy')

plt.subplot(2, 2, 4), plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')

plt.show()
