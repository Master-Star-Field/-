import cv2
import numpy as np
import matplotlib.pyplot as plt

def intrans(f, method, *args):
    # Verify the correct number of inputs.
    # if len(args) < 1 or len(args) > 3:
    #     raise ValueError("Incorrect number of inputs for the transformation method.")

    # Convert the input image to the range [0, 1] if necessary.
    if f.dtype == np.float64 and np.max(f) > 1 and method != 'log':
        f = cv2.normalize(f, None, 0, 1, cv2.NORM_MINMAX)

    # Perform the intensity transformation specified.
    if method == 'neg':
        g = cv2.bitwise_not(f)
    elif method == 'log':
        if len(args) == 0:
            c = 1
        elif len(args) == 1:
            c = args[0]
        elif len(args) == 2:
            c = args[0]
            classin = args[1]
        else:
            raise ValueError("Incorrect number of inputs for the log option.")
        g = c * np.log(1 + f.astype(np.float64))
    elif method == 'gamma':
        if len(args) < 1:
            raise ValueError("Not enough inputs for the gamma option.")
        gam = 3
        g = cv2.pow(f, gam)
    elif method == 'stretch':
        if len(args) == 0:
            m = np.mean(f)
            E = 1.0
        elif len(args) == 2:
            m = args[0]
            E = args[1]
        else:
            raise ValueError("Incorrect number of inputs for the stretch option.")
        g = 1 / (1 + (m / (f + np.finfo(float).eps))**E)
    else:
        raise ValueError("Unknown enhancement method.")

    # Convert the output to the same class as the input image.
    g = cv2.convertScaleAbs(g, alpha=(255.0 / np.max(g)))

    return g

f = cv2.imread('files/Fig0343.tif')
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

# Выполнение преобразования растяжения
g = intrans(gray, 'stretch')

# Создание фигуры и подуровней
fig, axs = plt.subplots(1, 4, figsize=(12, 4))

# Отображение исходного изображения
axs[0].imshow(f, cmap='gray')
axs[0].set_title('Original Image')

# Отображение преобразованного изображения с растяжением
axs[1].imshow(g, cmap='gray')
axs[1].set_title('Stretch')

# Отображение преобразованного изображения с гамма-коррекцией
axs[2].imshow(intrans(gray, 'gamma', 2), cmap='gray')
axs[2].set_title('Gamma')

# Отображение преобразованного изображения с логарифмическим преобразованием
axs[3].imshow(intrans(gray, 'log'), cmap='gray')
axs[3].set_title('Log')

# Удаление осей координат
for ax in axs:
    ax.axis('off')

# Сохранение изображения
plt.savefig("1.jpg")

# Отображение фигуры
plt.show()

