import cv2
import numpy as np
import matplotlib.pyplot as plt

# Путь к изображению
image_path = 'images/Fig0343.tif'

# Загрузка изображения
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = image.astype('double')

for i in image:
    i /= 255

# Размер маски фильтра и коэффициенты усиления высокочастотной составляющей
mask_size = 5  # Размер маски фильтра
gains = [2, 5, 0.5]  # Значения коэффициента усиления высокочастотной составляющей

kernel = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])

fig, axs = plt.subplots(len(gains) + 1, 1, figsize=(4, 12))

# Отображение исходного изображения
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Исходное изображение')

for i, gain in enumerate(gains):
    # Применение усредняющего фильтра
    blur = cv2.filter2D(image, -1, kernel)

    # Получение высокочастотной составляющей
    high_freq = image - blur

    # Повышение резкости изображения
    sharpened = image + gain * high_freq

    axs[i+1].imshow(sharpened, cmap='gray')
    axs[i+1].set_title(f'Gain = {gain}')
plt.tight_layout()
plt.show()
