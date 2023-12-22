import matplotlib.pyplot as plt
import numpy as np
import cv2

# Загрузка изображения пыльцы
image = cv2.imread('files/Fig0310.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow("name", image)
cv2.waitKey(0)
# Применение гистограммной эквализации
equ = cv2.equalizeHist(image)

# Вычисление гистограммы для исходного и эквализованного изображений
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_equ = cv2.calcHist([equ], [0], None, [256], [0, 256])

# Вычисление нормированной гистограммы для исходного и эквализованного изображений
hist_original_normalized = hist_original / np.sum(hist_original)
hist_equ_normalized = hist_equ / np.sum(hist_equ)

# Построение функции преобразования для гистограммной эквализации
cdf = hist_original.cumsum()
cdf_normalized = cdf * hist_original.max() / cdf.max()

# Визуализация результатов
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Исходное изображение')
cv2.imshow("name", equ)
cv2.waitKey(0)
plt.subplot(2, 2, 2)
plt.imshow(equ, cmap='gray')
plt.title('Изображение после гистограммной эквализации')

plt.subplot(2, 2, 3)
plt.plot(cdf_normalized, color='b')
plt.title('Функция преобразования')

plt.subplot(2, 2, 4)
plt.plot(hist_original_normalized, color='r', label='Исходное изображение')
plt.plot(hist_equ_normalized, color='g', label='После гистограммной эквализации')
plt.legend()
plt.title('Нормированные гистограммы')

plt.tight_layout()
plt.show()