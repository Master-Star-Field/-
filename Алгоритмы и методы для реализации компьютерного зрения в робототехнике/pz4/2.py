import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image = cv2.imread('trui.png', 0)

# Детектор LOG без фильтра
edges1 = cv2.Laplacian(image, cv2.CV_64F, ksize=5)

# Детектор Саппу с фильтром с сг = 6
edges2 = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)

# Детектор Саппу с фильтром с сг = 12
edges3 = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=7)

# Отображение результатов с помощью subplots
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Исходное изображение')

plt.subplot(2, 2, 2)
plt.imshow(edges1, cmap='gray')
plt.title('Детектор LOG без фильтра')

plt.subplot(2, 2, 3)
plt.imshow(edges2, cmap='gray')
plt.title('Детектор Саппу с фильтром с сг = 6')

plt.subplot(2, 2, 4)
plt.imshow(edges3, cmap='gray')
plt.title('Детектор Саппу с фильтром с сг = 12')

plt.tight_layout()
plt.show()
