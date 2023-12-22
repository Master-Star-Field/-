import matplotlib.pyplot as plt
import numpy as np
import cv2



image = cv2.imread('files/F1.tif', cv2.IMREAD_GRAYSCALE)

# Perform histogram equalization
equalized_image = cv2.equalizeHist(image)

# Define the parameters for the bimodal Gaussian function
m1 = 0.15
sig1 = 0.05
m2 = 0.75
sig2 = 0.05
A1 = 1
A2 = 0.07
k = 0.002

# Generate the bimodal Gaussian function
z = np.linspace(0, 1, 256)
cl = A1 * (1 / ((2 * np.pi) ** 0.5) * sig1)
kl = 2 * (sig1 ** 2)
c2 = A2 * (1 / ((2 * np.pi) ** 0.5) * sig2)
k2 = 2 * (sig2 ** 2)
p = k + cl * np.exp(-((z - m1) ** 2) / kl) + c2 * np.exp(-((z - m2) ** 2) / k2)
p = p / np.sum(p)

# Perform histogram matching
matched_image = cv2.LUT(equalized_image, p * 255).astype(np.uint8)

# Display the results
plt.figure()
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(122)
plt.imshow(equalized_image, cmap='gray')
plt.title('Histogram Equalization')

plt.show()
