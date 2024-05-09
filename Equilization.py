import cv2
import matplotlib.pyplot as plt

image_path = 'Lenna.png'

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

equ_img = cv2.equalizeHist(img)

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(equ_img, cmap='gray')
plt.title('Equalized Image')

plt.show()
