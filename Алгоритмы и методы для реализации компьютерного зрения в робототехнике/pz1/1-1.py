from PIL import Image
import os

#Задание 2

image_files = ["compressed_image.jpg", 'img.png', 'img_2bit.png']
for file in image_files:
    image_size = os.path.getsize(file) * 8
    image = Image.open(file)
    width, height = image.size
    pixel_mode = image.mode

    if pixel_mode == '1':
        pixel_depth = 1
    elif pixel_mode == 'L':
        pixel_depth = 8
    elif pixel_mode == 'RGB':
        pixel_depth = 24
    else:
        pixel_depth = None


    compression_ratio = (width * height * pixel_depth) / image_size


    print(f"Изображение: {file}")
    print(f"Размер: {width}x{height} пикселей")
    print(f"Разрядность пикселя: {pixel_depth}")
    print(f"Степень сжатия: {compression_ratio}\n")

#Задание 6
image = Image.open('pears.png').convert('L')

# Преобразуем изображение в двоичное с порогом 25%
binary_image_25 = image.point(lambda x: 0 if x < 64 else 255)

# Преобразуем изображение в двоичное с порогом 50%
binary_image_50 = image.point(lambda x: 0 if x < 128 else 255)

# Преобразуем изображение в двоичное с порогом 75%
binary_image_75 = image.point(lambda x: 0 if x < 192 else 255)

#binary_image_25.show()
#binary_image_50.show()
#binary_image_75.show()