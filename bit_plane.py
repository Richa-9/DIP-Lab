import cv2
import numpy as np

image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

num_bits = 8
bit_planes = []

for i in range(num_bits):
    bit_plane = np.bitwise_and(image, 1 << i)
    bit_planes.append(bit_plane)

for i in range(num_bits):
    cv2.imshow('Bit Plane ' + str(i), bit_planes[i] * 255) 
cv2.waitKey(0)  
cv2.destroyAllWindows()
