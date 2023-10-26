from PIL import Image
import os
import random
import cv2
import numpy as np
import scipy.io as sio
from show_mat import*



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
#


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
#
# cv2.imshow('res', img_2)
# cv2.waitKey(0)


"""--------------------------------- 4 ЗАДАЧА ----------------------------------------"""
# # #opencv
# arr = np.zeros((3, 4),dtype='uint8')
# print(arr)
# for i in range(0, 4):
#     for j in range(0, 3):
#         arr[j][i] = random.random()*21.0 - 10.5
# print(arr)
#
#
# cv2.imshow('res', arr)
# cv2.waitKey(0)


"""-------------------------------- 5 ЗАДАЧА -------------------------------------------"""

# show_mat("clown.mat")


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
#
# binary_image_25.show()
# binary_image_50.show()
# binary_image_75.show()

"""------------------------------------- 7 ЗАДАНИЕ ---------------------------------------------------"""


# Загрузка изображения
image = cv2.imread("peppers.png")
import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread("peppers.png")

# Конвертация в цветовое пространство HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Определение диапазона зеленого цвета в HSV
lower_green = np.array([20, 50, 50])
upper_green = np.array([80, 255, 255])

# Создание маски для зеленой компоненты
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# Инвертирование маски
inverted_mask = cv2.bitwise_not(green_mask)

# Применение инвертированной маски к исходному изображению
result = cv2.bitwise_and(image, image, mask=inverted_mask)

# Отображение результата
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""--------------------------------------- 8 ЗАДАНИЕ -------------------------------------------------"""

#
#
#
#
# # Загрузка видео
# video = cv2.VideoCapture('1.mp4')
#
# while True:
#     # Считывание кадра из видео
#     ret, frame = video.read()
#
#     if not ret:
#         break
#
#     # Преобразование кадра из RGB в HSV цветовое пространство
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     lower_red = np.array([0, 50, 50])
#     upper_red = np.array([10, 255, 255])
#     mask1 = cv2.inRange(hsv, lower_red, upper_red)
#
#     lower_red = np.array([80, 30, 30])
#     upper_red = np.array([180, 255, 255])
#     mask2 = cv2.inRange(hsv, lower_red, upper_red)
#
#
#     mask = cv2.bitwise_or(mask1, mask2)
#
#
#     result = cv2.bitwise_and(frame, frame, mask=mask2)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if w > 10 and h > 10 and y > 140:
#             cv2.putText(result, 'Red Object', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#             cv2.circle(result, (x + w // 2, y + h // 2), max(w, h) // 2, (0, 0, 255), 2)
#
#     cv2.imshow('Red Objects', result)
#
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#
# video.release()
# cv2.destroyAllWindows()
