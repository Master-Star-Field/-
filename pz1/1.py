#first test
import cv2
import numpy as np

# задание 1
img = cv2.imread('pears.png')
res_img = cv2.resize(img, (486, 732), cv2.INTER_NEAREST)

cv2.imshow('res', img)
cv2.waitKey(0)

print(img.shape)
cv2.imwrite('C:\с диска д вся инфа\img.png', img)


