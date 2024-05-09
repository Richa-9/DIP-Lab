import cv2
import numpy as np
import matplotlib.pyplot as plt

def pad_image(image, pad_size):
    padded_image = np.pad(image, pad_size, mode='constant')
    return padded_image

def roberts_operator(image):
    padded_image = pad_image(image, 1)

    kernel_x = np.array([[-1, 0], [0, 1]])

    kernel_y = np.array([[0, -1], [1, 0]])

    x = np.zeros_like(image)
    y = np.zeros_like(image)

    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            gx = np.sum(kernel_x * padded_image[i - 1:i + 1, j - 1:j + 1])
            gy = np.sum(kernel_y * padded_image[i - 1:i + 1, j - 1:j + 1])
            x[i - 1, j - 1] = gx
            y[i - 1, j - 1] = gy

    magnitude = np.sqrt(x ** 2 + y ** 2)

    return x, y, magnitude


image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

x, y, magnitude = roberts_operator(image)

plt.figure(figsize=(10,8))

plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 3, 2), plt.imshow(x, cmap='gray')
plt.title('gy')

plt.subplot(2, 3, 3), plt.imshow(y, cmap='gray')
plt.title('gx')

plt.subplot(2, 3, 4), plt.imshow(magnitude, cmap='gray')
plt.title('robert filter')

plt.show()
