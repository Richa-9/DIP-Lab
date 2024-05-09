import cv2
import numpy as np

# Load an image (replace 'your_image.jpg' with the path to your image)
input_image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

# Define a simple 3x3 kernel
kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

# Apply zero padding
padded_image = np.pad(input_image, ((1, 1), (1, 1)), mode='constant', constant_values=0)

# Create an empty image to store the correlated result
correlated_image = np.zeros_like(input_image)

# Apply correlation
for i in range(1, padded_image.shape[0] - 1):
    for j in range(1, padded_image.shape[1] - 1):
        neighborhood = padded_image[i-1:i+2, j-1:j+2]
        correlated_value = np.sum(neighborhood * kernel)
        correlated_image[i-1, j-1] = correlated_value

# Display the original and correlated images
cv2.imshow('Original Image', input_image)
cv2.imshow('Correlated Image', correlated_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
