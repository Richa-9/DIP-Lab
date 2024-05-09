import numpy as np
import cv2

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

def convolution(image, kernel):
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    img_padded = np.pad(image, pad_size, mode='constant')

    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(img_padded[i:i+kernel_size, j:j+kernel_size] * kernel)
    return result.astype(np.uint8)

img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)  

kernel_size = 5
sigma = 1.4

gaussian_kernel_filter = gaussian_kernel(kernel_size, sigma)
blurred_image = convolution(img, gaussian_kernel_filter)

cv2.imshow('Original Image', img)
cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)  
cv2.destroyAllWindows()  
