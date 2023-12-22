import numpy as np
import cv2
import matplotlib.pyplot as plt
#Задание 1
def imadjust(x,a,b,c,d,gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


image = cv2.imread("files/Fig0304.tif", cv2.COLOR_BGR2GRAY)
arr = np.asarray(image)
arr1=imadjust(arr,arr.min(),arr.max(),0,1)
arr2=imadjust(arr,arr.min(),arr.max(),1,0.3, 1)
arr3=imadjust(arr,arr.min(),arr.max(),0,1, 1.8)
arr4=imadjust(arr,arr.min(),arr.max(),0,1, 2.1)

arr1_image = np.uint8(arr1 * 255)
arr2_image = np.uint8(arr2 * 255)
arr3_image = np.uint8(arr3 * 255)
arr4_image = np.uint8(arr4 * 255)

opencv_image1 = cv2.cvtColor(arr1_image, cv2.COLOR_GRAY2BGR)
opencv_image2 = cv2.cvtColor(arr2_image, cv2.COLOR_GRAY2BGR)
opencv_image3 = cv2.cvtColor(arr3_image, cv2.COLOR_GRAY2BGR)
opencv_image4 = cv2.cvtColor(arr4_image, cv2.COLOR_GRAY2BGR)

hist_arr1 = cv2.calcHist([opencv_image1], [0], None, [256], [0, 256])
# Построение гистограммы и функции преобразования для arr2
hist_arr2 = cv2.calcHist([opencv_image2], [0], None, [256], [0, 256])
# Построение гистограммы и функции преобразования для arr3
hist_arr3 = cv2.calcHist([opencv_image3], [0], None, [256], [0, 256])
# Построение гистограммы и функции преобразования для arr4
hist_arr4 = cv2.calcHist([opencv_image4], [0], None, [256], [0, 256])

fig, axs = plt.subplots(4, 2, figsize=(10, 20))

# Построение изображения и гистограммы для arr1
axs[0, 0].imshow(arr1, cmap='gray')
axs[0, 0].set_title('Изображение arr1')
axs[0, 1].plot(hist_arr1, color='black')
axs[0, 1].set_title('Гистограмма arr1')

# Построение изображения и гистограммы для arr2
axs[1, 0].imshow(arr2, cmap='gray')
axs[1, 0].set_title('Изображение arr2')
axs[1, 1].plot(hist_arr2, color='black')
axs[1, 1].set_title('Гистограмма arr2')

# Построение изображения и гистограммы для arr3
axs[2, 0].imshow(arr3, cmap='gray')
axs[2, 0].set_title('Изображение arr3')
axs[2, 1].plot(hist_arr3, color='black')
axs[2, 1].set_title('Гистограмма arr3')

# Построение изображения и гистограммы для arr4
axs[3, 0].imshow(arr4, cmap='gray')
axs[3, 0].set_title('Изображение arr4')
axs[3, 1].plot(hist_arr4, color='black')
axs[3, 1].set_title('Гистограмма arr4')

# Отображение всех графиков
plt.tight_layout()
plt.show()