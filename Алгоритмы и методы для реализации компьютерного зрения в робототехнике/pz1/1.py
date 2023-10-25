# from PIL import Image
import os
import random
import cv2
import numpy as np
import scipy.io as sio

"""--------------------------------- 1 ЗАДАЧА ----------------------------------------"""
# img = cv2.imread('pears.png')
#
# cv2.imwrite('img.jpg', img)
# cv2.imwrite("compressed_image.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 10])
#
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# cv2.imwrite('img.png', gray)
#
# img_2bit = cv2.convertScaleAbs(img, alpha=(24.0/6.0))
# cv2.imwrite('img_2bit.png', img_2bit)



"""--------------------------------- 2 ЗАДАЧА ----------------------------------------"""

# image_files = ["compressed_image.jpg", 'img.png', 'img_2bit.png']
# for file in image_files:
#     image_size = os.path.getsize(file) * 8
#     image = Image.open(file)
#     width, height = image.size
#     pixel_mode = image.mode
#
#     if pixel_mode == '1':
#         pixel_depth = 1
#     elif pixel_mode == 'L':
#         pixel_depth = 8
#     elif pixel_mode == 'RGB':
#         pixel_depth = 24
#     else:
#         pixel_depth = None
#
#
#     compression_ratio = (width * height * pixel_depth) / image_size
#
#
#     print(f"Изображение: {file}")
#     print(f"Размер: {width}x{height} пикселей")
#     print(f"Разрядность пикселя: {pixel_depth}")
#     print(f"Степень сжатия: {compression_ratio}\n")



"""--------------------------------- 3 ЗАДАЧА ----------------------------------------"""
# img_1 = cv2.imread('2.png')
# img_2 = cv2.imread('2-1.png')
#
# # Первая картинка
# cv2.circle(img_1, (528, 418), 5, (255, 0, 0), -1)
# cv2.circle(img_1, (528, 278), 5, (255, 255, 0), -1)
#
# if img_1[278][528].item(0) > img_1[418][528].item(0):
#     res = img_1[278][528].item(0) - img_1[418][528].item(0)
#     liter_1 = 'A'
#     liter_2 = 'B'
# else:
#     res = img_1[418][528].item(0) - img_1[278][528].item(0)
#     liter_1 = 'B'
#     liter_2 = 'A'
#
# print('Первая картинка:')
# print(f'{liter_1} светлее {liter_2} на {res}\n')
#
# # Вторая картинка
# print('Вторая картинка:')
# print(f'A1: {img_2[328][208]}')
# print(f'A2: {img_2[378][228]}')
# print(f'B1: {img_2[328][778]}')
# print(f'B2: {img_2[378][798]}')
#
# gray_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)
#
# print(f'A1 - B1: {gray_2[328][208] - gray_2[328][778]}')
# print(f'A2 - B2: {gray_2[378][228] - gray_2[378][798]}')
# print('Обе пары точек не имеют между собой расхождения по яркости')

# cv2.imshow('res', img_2)
# cv2.waitKey(0)


"""--------------------------------- 4 ЗАДАЧА ----------------------------------------"""

# arr = np.zeros((6, 6))
# for i in range(0, 6):
#     for j in range(0, 6):
#         arr[j][i] = random.random()*21.0 - 10.5
# arr.astype(np.int32)

# cv2.imshow('res', arr)
# cv2.waitKey(0)


"""-------------------------------- 5 ЗАДАЧА -------------------------------------------"""

# clown = sio.loadmat('clown.mat')
# print(clown['X'][1][80])
#
# matrix = np.zeros(shape=(200, 320, 3), dtype=np.float32)
# print(matrix.shape)
#
# # clown_1 = clown['X'].reshape(64000, 1)
# # print(clown['X'][80][0])
# # print(clown_1[np.argmax(clown_1)])
# # print(clown['map'][80][1])
#
# for i in range(0, 199):
#     for j in range(0, 319):
#         m_1 = clown['X'][i][j]
#         print(m_1)
#         matrix[i][j][2] = clown['map'][ clown['X'][i][j] - 1][0]
#         matrix[i][j][1] = clown['map'][ clown['X'][i][j] - 1][1]
#         matrix[i][j][0] = clown['map'] [clown['X'][i][j] - 1][2]
#
# # matrix = cv2.cvtColor(matrix, cv2.COLOR_HSV2BGR)
#
# cv2.imshow('res', matrix)
# cv2.waitKey(0)


"""---------------------------------- 6 ЗАДАЧА ------------------------------------------------------"""
# image = Image.open('pears.png').convert('L')
#
# # Преобразуем изображение в двоичное с порогом 25%
# binary_image_25 = image.point(lambda x: 0 if x < 64 else 255)
#
# # Преобразуем изображение в двоичное с порогом 50%
# binary_image_50 = image.point(lambda x: 0 if x < 128 else 255)
#
# # Преобразуем изображение в двоичное с порогом 75%
# binary_image_75 = image.point(lambda x: 0 if x < 192 else 255)

#binary_image_25.show()
#binary_image_50.show()
#binary_image_75.show()

"""------------------------------------- 7 ЗАДАНИЕ ---------------------------------------------------"""

# img = cv2.imread('peppers.png')
# matrix = np.zeros((384, 512, 3), dtype='uint8')
#
# img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#
# for i in range(0,383):
#     for j in range(0, 511):
#         if img[i][j][0] < 95 and img[i][j][0] > 80:
#             matrix[i][j] = img[i][j]
#
# cv2.imshow('res', matrix)
# cv2.waitKey(0)


"""--------------------------------------- 8 ЗАДАНИЕ -------------------------------------------------"""

cap = cv2.VideoCapture("1.mp4")

while True:
    _, img = cap.read()

    matrix = np.zeros(img.shape, dtype='uint8')

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j][0] < 245 and img[i][j][0] > 100:
                matrix[i][j] = img[i][j]

    cv2.imshow('res', matrix)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break