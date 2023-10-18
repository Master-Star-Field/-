#first test
import cv2
import numpy as np

# задание 1
img = cv2.imread('pears.png')

cv2.imwrite('img.jpg', img)
cv2.imwrite("compressed_image.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 10])

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imwrite('img.png', gray)

img_2bit = cv2.convertScaleAbs(img, alpha=(24.0/6.0))
cv2.imwrite('img_2bit.png', img_2bit)





