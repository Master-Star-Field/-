import cv2

image = cv2.imread("images/Fig0107.tif")
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print(image.shape)

# Математическое ожидание
# n = 0
# s = 0

# for k in range(0, 255):
#     for j in range(0, 1838):
#         for i in range(0, 929):
#             if image[j][i] == k:
#                 n += 1
#
#     s += k*n/(1838*929)
#     n = 0

# Дисперсия
# for k in range(0, 255):
#     for j in range(0, 1838):
#         for i in range(0, 929):
#             if image[j][i] == k:
#                 n += 1
#
#     s += (k-149.75)*(k - 149.75)*n/(1838*929)
#     n = 0

sigma = 4845.929 ** 0.5
print(sigma)

image1 = cv2.GaussianBlur(image, (3, 3), 69.612)
cv2.imshow('orig', image)
cv2.imshow('res', image1)
cv2.waitKey(0)