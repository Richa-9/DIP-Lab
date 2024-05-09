import cv2
import numpy as np

def impulse_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)

    salt_pixels = np.random.rand(*image.shape) < salt_prob
    noisy_image[salt_pixels]=255

    pepper_pixels = np.random.rand(*image.shape) < pepper_prob
    noisy_image[pepper_pixels]=0

    return noisy_image

def box(image, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros_like(image)

    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            window = image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i+kernel_size//2, j+kernel_size//2] = np.median(window)

    return filtered_image

def adaptive_median_filter(image, Smax):
    height, width = image.shape
    filtered_image = np.zeros_like(image)

    padded_image = np.pad(image, Smax//2, mode='constant')

    for i in range(height):
        for j in range(width):
            window_size = 3
            while window_size <= Smax:
                window = padded_image[i:i+window_size, j:j+window_size]

                min_value = np.min(window)
                max_value = np.max(window)

                if min_value < image[i, j] < max_value:
                    filtered_pixel = box(window, 3)
                    filtered_image[i, j] = filtered_pixel[window_size//2, window_size//2]
                    break
                else:
                    window_size += 2

    return filtered_image

image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

salt_prob=0.1
pepper_prob=0.1
noisy_image = impulse_noise(image, salt_prob, pepper_prob)

Smax=7
filtered_image = adaptive_median_filter(noisy_image, Smax)

cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
