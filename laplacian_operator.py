import cv2
import numpy as np

def convolve(image, kernel):
    image_height, image_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    result = np.zeros_like(image)

    padded_image = np.zeros((image_height + pad_height * 2, image_width + pad_width * 2))
    padded_image[pad_height:padded_image.shape[0]-pad_height, pad_width:padded_image.shape[1]-pad_width] = image

    for y in range(image_height):
        for x in range(image_width):
            result[y, x] = np.sum(padded_image[y:y+kernel_height, x:x+kernel_width] * kernel)

    return result

def gaussian_blur(image, kernel_size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-kernel_size//2)**2 + (y-kernel_size//2)**2) / (2*sigma**2)), (kernel_size, kernel_size))

    kernel /= np.sum(kernel)
    return convolve(image, kernel)

def laplacian_filter(image):
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return convolve(image, laplacian_kernel)

def laplacian_gaussian(img):
    img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
    blurred = gaussian_blur(img, kernel_size=3, sigma=1.0)

    sharpened = laplacian_filter(blurred)

    cv2.imshow('Original', img)
    cv2.imshow('Blurred', blurred)
    cv2.imshow('Sharpened', sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

laplacian_gaussian('Lenna.png')
