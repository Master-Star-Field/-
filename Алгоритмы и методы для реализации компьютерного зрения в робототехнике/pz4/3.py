import cv2
import numpy as np


def kmeans_mono():
    # Загрузка монохромного изображения
    image = cv2.imread('pic.png', 0)

    # Преобразование изображения в одномерный массив
    data = image.reshape((-1, 1))

    # Преобразование в тип данных с плавающей запятой
    data = np.float32(data)

    # Определение критериев останова для алгоритма к-средних
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Применение алгоритма к-средних
    _, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Конвертация центров кластеров обратно в тип данных с плавающей запятой
    centers = np.uint8(centers)

    # Преобразование меток обратно в изображение
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    # Вывод центров кластеров
    print('Центры кластеров для монохромного изображения:')
    print(centers)

    # Отображение исходного и сегментированного изображений
    cv2.imshow('Original Image', image)
    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Сегментация цветного изображения kobi.png на два и четыре региона
def kmeans_color():
    # Загрузка цветного изображения
    image = cv2.imread('pic.png')

    # Преобразование изображения в одномерный массив
    data = image.reshape((-1, 3))

    # Преобразование в тип данных с плавающей запятой
    data = np.float32(data)

    # Определение критериев останова для алгоритма к-средних
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Применение алгоритма к-средних для двух регионов
    _, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Применение алгоритма к-средних для четырех регионов
    _, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Конвертация центров кластеров обратно в тип данных с плавающей запятой
    centers2 = np.uint8(centers2)
    centers4 = np.uint8(centers4)

    # Преобразование меток обратно в изображение
    segmented_image2 = centers2[labels2.flatten()]
    segmented_image2 = segmented_image2.reshape(image.shape)

    segmented_image4 = centers4[labels4.flatten()]
    segmented_image4 = segmented_image4.reshape(image.shape)

    # Вывод центров кластеров
    print('Центры кластеров для цветного изображения (два региона):')
    print(centers2)
    print('Центры кластеров для цветного изображения (четыре региона):')
    print(centers4)

    # Отображение исходного и сегментированного изображений
    cv2.imshow('Original Image', image)
    cv2.imshow('Segmented Image (2 clusters)', segmented_image2)
    cv2.imshow('Segmented Image (4 clusters)', segmented_image4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#kmeans_mono()
kmeans_color()