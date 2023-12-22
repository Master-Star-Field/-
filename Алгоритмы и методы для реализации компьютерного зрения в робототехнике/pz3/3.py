import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('images/cameraman.tif', cv2.IMREAD_UNCHANGED)

def gaussian_filter(img, K_size=3, sigma=1.3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float64)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float64)

    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float64)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()

    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out

# Define the values
values = [
    ([50, 50], 9),
    ([50, 50], 1),
    ([9, 9], 9),
    ([9, 9], 50),
    ([3, 3], 50)
]

# Create subplots
fig, axs = plt.subplots(len(values))

# Apply Gaussian filter for each value and display the result
for i, (k_size, sigma) in enumerate(values):
    out = gaussian_filter(img, K_size=k_size[0], sigma=sigma)

    axs[i].imshow(out, cmap='gray')
    #axs[i].set_title(f'K_size={k_size[0]}, sigma={sigma}', loc='left')

    # Set aspect ratio to 'auto' to display images in their original resolution
    #axs[i].set_aspect('auto')

    # Set the y-axis label on the left side of the image
    axs[i].yaxis.set_label_position("left")
    axs[i].set_ylabel(f'K_size={k_size[0]}, sigma={sigma}')

fig.set_size_inches(200, 200)
# Increase the size of the images
plt.tight_layout()
'''
for ax in axs:

    ax.set_aspect(1.0/ax.get_data_ratio()/6, adjustable='box')
'''
# Увеличение размера изображений
fig.set_size_inches(6, 12)
# Display the subplot
plt.show()
