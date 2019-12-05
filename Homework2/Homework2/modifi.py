import numpy as np
import cv2
import os

def change_npy_to_txt():
    for index in range(1, 3):
        points_2d = np.load('Point2D_%d.npy' % index)

        with open('Point2D_%d.txt' % index, 'wb') as f:
            for pindex in range(points_2d.shape[0]):
                x, y = tuple(points_2d[pindex, :])
                str_to_write = '%d %d\r\n' % (int(x), int(y))
                f.write(str_to_write.encode('utf-8'))


def read_2d_points(path):
    if not os.path.exists(path):
        return None
    pylist_points = []
    with open(path, 'r', encoding='utf-8') as f:
        while 1:
            recv = f.readline()
            if recv == '':
                break
            x, y = (int(num) for num in recv.split(' '))
            pylist_points.append([x, y])
    return np.asarray(pylist_points)

a = read_2d_points('Point2D_2.txt')
np.save('Point2D_2', a)

img = cv2.imread('data/chessboard_2.jpg')
for i in range(36):
    x, y = tuple(a[i, :])
    cv2.circle(img, (np.int(x), np.int(y)), 5, (0, 255, 255))

cv2.imwrite('chessboard_points_2.png', img)