import cv2
import numpy as np


def apply_geometric_mean_filter(image, kernel_size):
    # Get image dimensions
    height, width = image.shape

    # Initialize filtered image
    filtered_image = np.zeros_like(image, dtype=np.float32)

    # Pad the image
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)),
                          mode='constant')

    # Define the range of the filter
    half_kernel = kernel_size // 2

    # Apply geometric mean filter
    for i in range(half_kernel, height + half_kernel):
        for j in range(half_kernel, width + half_kernel):
            # Extract the region around the pixel
            region = padded_image[i - half_kernel:i + half_kernel + 1, j - half_kernel:j + half_kernel + 1]

            # Compute the geometric mean
            geometric_mean = np.prod(region) ** (1 / (kernel_size ** 2))

            # Update the pixel value in the filtered image
            filtered_image[i - half_kernel, j - half_kernel] = geometric_mean

    return np.uint8(filtered_image)


# Load an image
image_path = "Lenna.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the size of the filter (must be odd)
kernel_size = 3

# Apply geometric mean filter
filtered_image = apply_geometric_mean_filter(image, kernel_size)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
