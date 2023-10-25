def show_mat(image):
    from scipy.io import loadmat
    from PIL import Image

    data = loadmat(image)
    pixels = data .get('X')
    colors = data['map']

    dict_colors = {i+1: colors[i] for i in range(len(colors))}

    width = len(pixels[0])
    height = len(pixels)

    image = Image.new("RGB", (width, height))
    res = image.load()

    for i in range(width):
        for j in range(height):
            c = dict_colors[pixels[j][i]]
            res[i, j] = (int(255 * c[0]), int(255 * c[1]), int(255 * c[2]))
    image.show()