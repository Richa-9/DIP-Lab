import cv2
import numpy as np

input_image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.uint8)
def erosion(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape

    result = np.zeros((height, width), dtype=np.uint8)

    for i in range(k_height // 2, height - k_height // 2):
        for j in range(k_width // 2, width - k_width // 2):
            intersection = image[i - k_height // 2:i + k_height // 2 + 1, j - k_width // 2:j + k_width // 2 + 1]
            result[i, j] = np.min(intersection * kernel)

    return result


def dilation(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape

    result = np.zeros((height, width), dtype=np.uint8)

    for i in range(k_height // 2, height - k_height // 2):
        for j in range(k_width // 2, width - k_width // 2):
            intersection = image[i - k_height // 2:i + k_height // 2 + 1, j - k_width // 2:j + k_width // 2 + 1]
            result[i, j] = np.max(intersection * kernel)

    return result

eroded_image = erosion(input_image, kernel)
cv2.imshow('Eroded',eroded_image)
dilated_image = dilation(input_image, kernel)
cv2.imshow('Dilated',dilated_image)
opening_image = dilation(eroded_image, kernel)
cv2.imshow('Opened',opening_image)
cv2.imwrite('Opened Image.png',opening_image)
cv2.waitKey(0)