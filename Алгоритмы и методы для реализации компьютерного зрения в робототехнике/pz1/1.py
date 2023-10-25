#first test
import random
import cv2
import numpy as np
import scipy.io as sio

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

# cv2.imshow('res', img_2)
# cv2.waitKey(0)


"""--------------------------------- 4 ЗАДАЧА ----------------------------------------"""

arr = np.zeros((6, 6))
for i in range(0, 6):
    for j in range(0, 6):
        arr[j][i] = random.random()*21.0 - 10.5
arr.astype(np.int32)

# cv2.imshow('res', arr)
# cv2.waitKey(0)

"""-------------------------------- 5 ЗАДАЧА -------------------------------------------"""

clown = sio.loadmat('clown.mat')
print(clown['X'][1][80])

matrix = np.zeros(shape=(200, 320, 3), dtype=np.float32)
print(matrix.shape)

# clown_1 = clown['X'].reshape(64000, 1)
# print(clown['X'][80][0])
# print(clown_1[np.argmax(clown_1)])
# print(clown['map'][80][1])

for i in range(0, 199):
    for j in range(0, 319):
        m_1 = clown['X'][i][j]
        print(m_1)
        matrix[i][j][2] = clown['map'][ clown['X'][i][j] - 1][0]
        matrix[i][j][1] = clown['map'][ clown['X'][i][j] - 1][1]
        matrix[i][j][0] = clown['map'] [clown['X'][i][j] - 1][2]

# matrix = cv2.cvtColor(matrix, cv2.COLOR_HSV2BGR)

cv2.imshow('res', matrix)
cv2.waitKey(0)