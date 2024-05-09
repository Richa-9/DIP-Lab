import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("Lenna.png")
image = cv2.resize(image, (500,500))
#cv2.imshow("original", image)
#cv2.waitKey(0)
s = image.shape

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('binary',imageGray)
cv2.waitKey(0)
h = np.zeros(shape=(256,1))
print(h)
for i in range(s[0]):
    for j in range(s[1]):
        k=imageGray[i,j]
        h[k,0]=h[k,0]+1
print(h)
plt.plot(h)
plt.show()
