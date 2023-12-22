import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image = cv2.imread('images/Fig0335.tif')

# Применение медианного фильтра с разными размерами маски фильтра
filtered_images = []
kernel_sizes = [1, 3, 5]  # Размеры маски фильтра

fig, axes = plt.subplots(len(kernel_sizes) + 1, 1, figsize=(4, 12))

# Отображение исходного изображения
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')

for i, kernel_size in enumerate(kernel_sizes):
    filtered_image = cv2.medianBlur(image, kernel_size)
    filtered_images.append(filtered_image)
    axes[i+1].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    axes[i+1].set_title(f'Mask size: {kernel_size}')

plt.tight_layout()
plt.show()