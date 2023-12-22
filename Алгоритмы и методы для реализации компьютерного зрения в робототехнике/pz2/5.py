from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2



def enhance_image_with_haze(image_path):
    # Загрузка изображения
    img = cv2.imread(image_path)

    # Применение алгоритма для улучшения отображения с туманом
    enhanced_image = cv2.detailEnhance(img)
    hist, bins = np.histogram(enhanced_image, 256)
    # calculate cdf
    cdf = hist.cumsum()
    # plot hist

    # remap cdf to [0,255]
    cdf = (cdf - cdf[0]) * 255 / (cdf[-1] - 1)
    cdf = cdf.astype(np.uint8)  # Transform from float64 back to unit8

    img2 = np.zeros((384, 495, 1), dtype=np.uint8)
    img2 = cdf[img]
    # Создание подложки для графика
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Отображение изначального изображения
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Изначальное изображение")

    # Отображение обработанного изображения
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Обработанное изображение")

    # Удаление осей координат
    for ax in axes:
        ax.axis("off")

    # Отображение графика
    plt.show()


# Путь к изображению с туманом
image_path = "files/fog.jpg"

# Вызов функции для улучшения отображения
enhance_image_with_haze(image_path)

img = cv2.imread('files/seabed.png')

# calculate hist
hist, bins = np.histogram(img, 256)
# calculate cdf
cdf = hist.cumsum()
# plot hist
plt.plot(hist,'r')

# remap cdf to [0,255]
cdf = (cdf-cdf[0])*255/(cdf[-1]-1)
cdf = cdf.astype(np.uint8)# Transform from float64 back to unit8

# generate img after Histogram Equalization
img2 = np.zeros((384, 495, 1), dtype =np.uint8)
img2 = cdf[img]

hist2, bins2 = np.histogram(img2, 256)
cdf2 = hist2.cumsum()
plt.plot(hist2, 'g')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Отображение изначального изображения
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Изначальное изображение")

# Отображение обработанного изображения
axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[1].set_title("Обработанное изображение")

# Удаление осей координат
for ax in axes:
    ax.axis("off")

# Отображение графика
plt.show()