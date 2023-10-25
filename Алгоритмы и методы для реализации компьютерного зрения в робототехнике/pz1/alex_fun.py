def fun_alex(image):

    import numpy as np
    import cv2
    import scipy.io as sio

    clown = sio.loadmat(image)
    print(clown['X'][1][80])

    matrix = np.zeros(shape=(200, 320, 3), dtype=np.float32)
    print(matrix.shape)


    for i in range(0, 199):
        for j in range(0, 319):
            m_1 = clown['X'][i][j]
            matrix[i][j][2] = clown['map'][clown['X'][i][j] - 1][0]
            matrix[i][j][1] = clown['map'][clown['X'][i][j] - 1][1]
            matrix[i][j][0] = clown['map'][clown['X'][i][j] - 1][2]


    cv2.imshow('res', matrix)
    #cv2.waitKey(0)