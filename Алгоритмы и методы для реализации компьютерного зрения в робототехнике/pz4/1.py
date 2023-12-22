import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image = cv2.imread('coins.png', 0)

# Полиномиальная аппроксимация
blur = cv2.GaussianBlur(image, (5, 5), 0)
_, th1 = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

# Метод Отсу
_, th2 = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

# Метод автоматического определения порога
th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 20)

# Отображение результатов с помощью subplots
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Исходное изображение')

plt.subplot(2, 2, 2)
plt.imshow(th1, cmap='gray')
plt.title('Polynomial Approximation')

plt.subplot(2, 2, 3)
plt.imshow(th2, cmap='gray')
plt.title('Otsu Method')

plt.subplot(2, 2, 4)
plt.imshow(th3, cmap='gray')
plt.title('Automatic Threshold')
plt.tight_layout()
plt.show()
