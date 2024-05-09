import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_padding(image, padding_size):
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    return padded_image

def convolve(image, kernel):
    return np.abs(np.convolve(image.flatten(), kernel.flatten(), mode='same').reshape(image.shape))

def sobel_filter(image, filter_x, filter_y):
    padded_image = apply_padding(image, padding_size=1)

    gx = convolve(padded_image, filter_x)
    gy = convolve(padded_image, filter_y)
    return gx, gy

image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

filter_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

filter_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

gx, gy = sobel_filter(image, filter_x, filter_y)

gradient_combined = gx + gy

plt.figure(figsize=(15, 15))

plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 3, 2), plt.imshow(gx, cmap='gray')
plt.title('gx')

plt.subplot(2, 3, 3), plt.imshow(gy, cmap='gray')
plt.title('gy')

plt.subplot(2, 3, 4), plt.imshow(gradient_combined, cmap='gray')
plt.title('gx + gy')

plt.show()

