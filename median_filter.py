import cv2
import numpy as np

image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

neighbor_size = 3

padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
filtered_image = np.zeros_like(image)

for i in range(neighbor_size // 2, padded_image.shape[0] - neighbor_size // 2):
    for j in range(neighbor_size // 2, padded_image.shape[1] - neighbor_size // 2):
        neighbor = padded_image[i - neighbor_size // 2:i + neighbor_size // 2 + 1, j - neighbor_size // 2:j + neighbor_size // 2 + 1]

        filtered_image[i - neighbor_size // 2, j - neighbor_size // 2] = np.median(neighbor)

cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
