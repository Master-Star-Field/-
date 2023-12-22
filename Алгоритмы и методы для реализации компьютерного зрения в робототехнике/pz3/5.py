import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image = cv2.imread('images/Fig0343.tif', cv2.IMREAD_GRAYSCALE)

a_values = [0, 0.5, 1]  # Значения переменной a

fig, axs = plt.subplots(len(a_values), 1, figsize=(6, 12))

for i, a in enumerate(a_values):
    kernel = np.array([[a/(a + 1), (1 - a)/(a + 1), a/(a + 1)],
                       [(1 - a)/(a + 1), -4/(a + 1), (1-a)/(a + 1)],
                       [a/(1 + a), (1-a)/(a + 1), a/(a + 1)]])

    sharp_img = image - 0.5*cv2.filter2D(image, -1, kernel)

    axs[i].imshow(sharp_img, cmap='gray')
    axs[i].set_title(f'a = {a}')

plt.tight_layout()
plt.show()
