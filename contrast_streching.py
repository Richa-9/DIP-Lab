import cv2
import numpy as np

image_path = "Lenna.png"
image = cv2.imread(image_path)

min_value = np.min(image)
max_value = np.max(image)

stretched_image = np.clip(image, min_value, max_value)

stretched_image = ((stretched_image - min_value) / (max_value - min_value)) * 255

stretched_image = np.uint8(stretched_image)

cv2.imshow("Original Image", image)
cv2.imshow("Contrast-Stretched Image", stretched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
