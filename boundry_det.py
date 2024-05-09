import cv2
import numpy as np

input_image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

# Define a kernel for erosion and dilation
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

def boundary_extraction(image,eroded_image):
    boundary_extracted_img = image - eroded_image
    return boundary_extracted_img

eroded_image = erosion(input_image, kernel)
cv2.imshow('Eroded',eroded_image)
boundary_img = boundary_extraction(input_image,eroded_image)
cv2.imshow('Boundary Extracted',boundary_img)
cv2.imwrite('Boundary Extracted Image.png',boundary_img)
cv2.waitKey(0)