from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

Задание 1
def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


image = Image.open("files/Fig0304.tif")
arr = np.asarray(image)
arr2=imadjust(arr,arr.min(),arr.max(),0,1)

fig = plt.figure()
fig.suptitle('image')
plt.imshow(arr2)
plt.show()

image = cv2.imread('files/Fig0304.tif', 0)

# Преобразование изображения типа 1
image_type1 = cv2.equalizeHist(image)

# Преобразование изображения типа 2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
image_type2 = clahe.apply(image)

# Преобразование изображения типа 3
image_type3 = cv2.equalizeHist(image)
image_type3 = cv2.GaussianBlur(image_type3, (5, 5), 0)

# Преобразование изображения типа 4
image_type4 = cv2.equalizeHist(image)
image_type4 = cv2.medianBlur(image_type4, 5)

# Построение гистограммы для каждого изображения
hist_type1 = cv2.calcHist([image_type1], [0], None, [256], [0, 256])
hist_type2 = cv2.calcHist([image_type2], [0], None, [256], [0, 256])
hist_type3 = cv2.calcHist([image_type3], [0], None, [256], [0, 256])
hist_type4 = cv2.calcHist([image_type4], [0], None, [256], [0, 256])

# Построение функции преобразования для каждого изображения
cdf_type1 = hist_type1.cumsum()
cdf_type1_normalized = cdf_type1 * hist_type1.max() / cdf_type1.max()

cdf_type2 = hist_type2.cumsum()
cdf_type2_normalized = cdf_type2 * hist_type2.max() / cdf_type2.max()

cdf_type3 = hist_type3.cumsum()
cdf_type3_normalized = cdf_type3 * hist_type3.max() / cdf_type3.max()

cdf_type4 = hist_type4.cumsum()
cdf_type4_normalized = cdf_type4 * hist_type4.max() / cdf_type4.max()

# Визуализация результатов
plt.figure(figsize=(12, 8))

plt.subplot(2, 4, 1)

plt.imshow(image_type1, cmap='gray')
plt.title('Изображение типа 1')

plt.subplot(2, 4, 2)
plt.imshow(image_type2, cmap='gray')
plt.title('Изображение типа 2')

plt.subplot(2, 4, 3)
plt.imshow(image_type3, cmap='gray')
plt.title('Изображение типа 3')

plt.subplot(2, 4, 4)
plt.imshow(image_type4, cmap='gray')
plt.title('Изображение типа 4')

plt.subplot(2, 4, 5)
plt.plot(cdf_type1_normalized, color='b')
plt.title('Гистограмма изображения типа 1')

plt.subplot(2, 4, 6)
plt.plot(cdf_type2_normalized, color='b')
plt.title('Гистограмма изображения типа 2')

plt.subplot(2, 4, 7)
plt.plot(cdf_type3_normalized, color='b')
plt.title('Гистограмма изображения типа 3')

plt.subplot(2, 4, 8)
plt.plot(cdf_type4_normalized, color='b')
plt.title('Гистограмма изображения типа 3')

plt.tight_layout()
plt.show()


#Задание 2
#
# # Загрузка изображения пыльцы
# image = cv2.imread('files/Fig0310.tif', 0)
#
# # Применение гистограммной эквализации
# equ = cv2.equalizeHist(image)
#
# # Вычисление гистограммы для исходного и эквализованного изображений
# hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
# hist_equ = cv2.calcHist([equ], [0], None, [256], [0, 256])
#
# # Вычисление нормированной гистограммы для исходного и эквализованного изображений
# hist_original_normalized = hist_original / np.sum(hist_original)
# hist_equ_normalized = hist_equ / np.sum(hist_equ)
#
# # Построение функции преобразования для гистограммной эквализации
# cdf = hist_original.cumsum()
# cdf_normalized = cdf * hist_original.max() / cdf.max()
#
# # Визуализация результатов
# plt.figure(figsize=(12, 8))
#
# plt.subplot(2, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Исходное изображение')
#
# plt.subplot(2, 2, 2)
# plt.imshow(equ, cmap='gray')
# plt.title('Изображение после гистограммной эквализации')
#
# plt.subplot(2, 2, 3)
# plt.plot(cdf_normalized, color='b')
# plt.title('Функция преобразования')
#
# plt.subplot(2, 2, 4)
# plt.plot(hist_original_normalized, color='r', label='Исходное изображение')
# plt.plot(hist_equ_normalized, color='g', label='После гистограммной эквализации')
# plt.legend()
# plt.title('Нормированные гистограммы')
#
# plt.tight_layout()
# plt.show()

#Задание 3.
# def intrans(f, method, *args):
#     # Convert the input image to a numpy array
#     f = np.array(f)
#
#     # Perform the intensity transformation specified
#     if method == 'neg':
#         g = cv2.bitwise_not(f)
#     elif method == 'log':
#         if len(args) == 0:
#             c = 1
#         elif len(args) == 1:
#             c = args[0]
#         elif len(args) == 2:
#             c = args[0]
#             classin = args[1]
#         else:
#             raise ValueError('Incorrect number of inputs for the log option.')
#         g = c * np.log(1 + f.astype(float))
#     elif method == 'gamma':
#         if len(args) < 1:
#             raise ValueError('Not enough inputs for the gamma option.')
#         gam = args[0]
#         g = cv2.pow(f, gam)
#     elif method == 'stretch':
#         if len(args) == 0:
#             m = np.mean(f)
#             E = 4.0
#         elif len(args) == 2:
#             m = args[0]
#             E = args[1]
#         else:
#             raise ValueError('Incorrect number of inputs for the stretch option.')
#         g = 1 / (1 + (m / (f + np.finfo(float).eps)) ** E)
#     else:
#         raise ValueError('Unknown enhancement method.')
#
#     # Convert the output image to the same class as the input image
#     g = g.astype(f.dtype)
#
#     return g
#
#
# image = cv2.imread('files/Fig0343.tif', cv2.IMREAD_GRAYSCALE)
#
# # Perform the intensity transformation using the 'scratch' method
# transformed_image = intrans(image, 'stretch')
#
# # Display the original and transformed images
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
#
# plt.subplot(1, 2, 2)
# plt.imshow(transformed_image, cmap='gray')
# plt.title('Transformed Image')
#
# plt.show()

# Load the input image
#Задание 4
# image = cv2.imread('files/F1.tif', cv2.IMREAD_GRAYSCALE)
#
# # Perform histogram equalization
# equalized_image = cv2.equalizeHist(image)
#
# # Define the parameters for the bimodal Gaussian function
# m1 = 0.15
# sig1 = 0.05
# m2 = 0.75
# sig2 = 0.05
# A1 = 1
# A2 = 0.07
# k = 0.002
#
# # Generate the bimodal Gaussian function
# z = np.linspace(0, 1, 256)
# cl = A1 * (1 / ((2 * np.pi) ** 0.5) * sig1)
# kl = 2 * (sig1 ** 2)
# c2 = A2 * (1 / ((2 * np.pi) ** 0.5) * sig2)
# k2 = 2 * (sig2 ** 2)
# p = k + cl * np.exp(-((z - m1) ** 2) / kl) + c2 * np.exp(-((z - m2) ** 2) / k2)
# p = p / np.sum(p)
#
# # Perform histogram matching
# matched_image = cv2.LUT(equalized_image, p * 255).astype(np.uint8)
#
# # Display the results
# plt.figure()
# plt.subplot(131)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
# plt.subplot(132)
# plt.imshow(equalized_image, cmap='gray')
# plt.title('Histogram Equalization')
# plt.subplot(133)
# plt.imshow(matched_image, cmap='gray')
# plt.title('Histogram Matching')
# plt.show()

#Задание 5


def enhance_image_with_haze(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Применение алгоритма для улучшения отображения с туманом
    enhanced_image = cv2.detailEnhance(image)

    # Создание подложки для графика
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Отображение изначального изображения
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Изначальное изображение")

    # Отображение обработанного изображения
    axes[1].imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
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

def enhance_image_with_clouds(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Увеличение контрастности изображения
    enhanced_image = cv2.convertScaleAbs(image, alpha=1.1, beta=0.9)

    # Создание подложки для графика
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Отображение изначального изображения
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Изначальное изображение")

    # Отображение обработанного изображения
    axes[1].imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Обработанное изображение")

    # Удаление осей координат
    for ax in axes:
        ax.axis("off")

    # Отображение графика
    plt.show()


image_path = "files/seabed.png"
enhance_image_with_clouds(image_path)