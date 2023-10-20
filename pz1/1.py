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



















"""--------------------------------- 3 ЗАДАЧА ----------------------------------------"""
img_1 = cv2.imread('2.png')
img_2 = cv2.imread('2-1.png')

# Первая картинка
cv2.circle(img_1, (528, 418), 5, (255, 0, 0), -1)
cv2.circle(img_1, (528, 278), 5, (255, 255, 0), -1)

if img_1[278][528].item(0) > img_1[418][528].item(0):
    res = img_1[278][528].item(0) - img_1[418][528].item(0)
    liter_1 = 'A'
    liter_2 = 'B'
else:
    res = img_1[418][528].item(0) - img_1[278][528].item(0)
    liter_1 = 'B'
    liter_2 = 'A'

print('Первая картинка:')
print(f'{liter_1} светлее {liter_2} на {res}\n')

# Вторая картинка
print('Вторая картинка:')
print(f'A1: {img_2[328][208]}')
print(f'A2: {img_2[378][228]}')
print(f'B1: {img_2[328][778]}')
print(f'B2: {img_2[378][798]}')

gray_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)

print(f'A1 - B1: {gray_2[328][208] - gray_2[328][778]}')
print(f'A2 - B2: {gray_2[378][228] - gray_2[378][798]}')
print('Обе пары точек не имеют между собой расхождения по яркости')

cv2.imshow('res', img_2)
cv2.waitKey(0)


