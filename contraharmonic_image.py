import numpy as np
import cv2

def contraharmonic_filter(image, Q):
    height, width = image.shape

    filtered_image = np.zeros_like(image, dtype=np.float32)

    padded_image = np.pad(image, ((3 // 2, 3 // 2), (3 // 2, 3 // 2)),
                          mode='constant')

    half_kernel = 3 // 2

    for i in range(half_kernel, height + half_kernel):
        for j in range(half_kernel, width + half_kernel):
            region = padded_image[i - half_kernel:i + half_kernel + 1, j - half_kernel:j + half_kernel + 1]

            numerator = np.sum(np.power(region, Q + 1))
            denominator = np.sum(np.power(region, Q))

            if denominator == 0:
                filtered_image[i - half_kernel, j - half_kernel] = 0
            else:
                filtered_image[i - half_kernel, j - half_kernel] = numerator / denominator

    return np.uint8(filtered_image)

image_path = "Lenna.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

Q = 1.5

filtered_image = contraharmonic_filter(image, Q)

cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
