import cv2
import numpy as np

input_image = cv2.imread('Lenna.png')

gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

avg = np.ones((3, 3), np.float32) / 9

filtered_image = cv2.filter2D(gray_image, -1, avg)

cv2.imshow('Original Image', gray_image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
