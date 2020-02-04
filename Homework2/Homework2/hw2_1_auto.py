import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.linalg
from scipy import signal

def read_3d_points(path):
    if not os.path.exists(path):
        return None
    pylist_points = []
    with open(path, 'r', encoding='utf-8') as f:
        while 1:
            recv = f.readline()
            if recv == '':
                break
            x, y, z = (int(num) for num in recv.split(' '))
            pylist_points.append([x, y, z])
    return np.asarray(pylist_points)


def calibrate(P2, P3):
    nums = P2.shape[0]

    A = np.zeros((2*nums, 12), dtype=np.float64)
    h = 0
    for index in range(nums):
        x, y = tuple(P2[index, :2])
        A[h, :] = np.array([P3[index, 0], P3[index, 1], P3[index, 2], 1,
                            0, 0, 0, 0,
                            -x * P3[index, 0], -x * P3[index, 1], -x * P3[index, 2], -x])
        A[h+1, :] = np.array([0, 0, 0, 0,
                              P3[index, 0], P3[index, 1], P3[index, 2], 1,
                              -y * P3[index, 0], -y * P3[index, 1], -y * P3[index, 2], -y])
        h += 2

    U, S, V = np.linalg.svd(A)
    P = V.T[:, 11]
    P = P.reshape((3, 4))
    return P


def plot_3d_camera(ax, P3, R3, color, text):
    r1 = R3.T @ [-0.5, 0.5, -0.25] + P3
    r2 = R3.T @ [0.5, 0.5, -0.25] + P3
    r3 = R3.T @ [-0.5, -0.5, -0.25] + P3
    r4 = R3.T @ [0.5, -0.5, -0.25] + P3
    r5 = R3.T @ [-0.5, 0.5, 0.25] + P3
    r6 = R3.T @ [0.5, 0.5, 0.25] + P3
    r7 = R3.T @ [-0.5, -0.5, 0.25] + P3
    r8 = R3.T @ [0.5, -0.5, 0.25] + P3

    verts = []
    verts.append([r1, r2, r4, r3])
    verts.append([r5, r6, r8, r7])
    verts.append([r1, r2, r6, r5])
    verts.append([r3, r4, r8, r7])
    verts.append([r1, r3, r7, r5])
    verts.append([r2, r4, r8, r6])

    poly = Poly3DCollection(verts, facecolor=color, edgecolors=['k'])
    poly.set_alpha(0.5)
    ax.add_collection(poly)
    line = np.array([P3, P3 + 5 * R3.T[:, 2]])
    ax.plot(line[:, 0], line[:, 1], zs=line[:, 2], color='k', linewidth=3)
    ax.text(P3[0]-0.7, P3[1]-0.7, P3[2]+1.4, text)


def plot_3d_scene(fig, cam1, R1, cam2, R2):
    ax = Axes3D(fig)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-10, 10])
    ax.set_ylim([10, -10])
    ax.set_zlim([-10, 10])

    # plot xy plane
    mode = 'w'
    alpha = 0.3
    for y in range(0, -4, -1):
        for x in range(0, 3):
            verts = [np.array([[x, y, 0], [x+1, y, 0], [x+1, y-1, 0], [x, y-1, 0]])]
            ax.add_collection(Poly3DCollection(verts, facecolors=mode, edgecolors=['k'], alpha=alpha))
            if mode == 'w':
                mode = 'k'
                alpha = 0.95
            else:
                mode = 'w'
                alpha = 0.3

    # plot xz plane
    mode = 'w'
    alpha = 0.3
    for z in range(0, -4, -1):
        for x in range(0, 3):
            verts = [np.array([[x, 0, z], [x+1, 0, z], [x+1, 0, z-1], [x, 0, z-1]])]
            ax.add_collection(Poly3DCollection(verts, facecolors=mode, edgecolors=['k'], alpha=alpha))
            if mode == 'w':
                mode = 'k'
                alpha = 0.95
            else:
                mode = 'w'
                alpha = 0.3

    plot_3d_camera(ax, cam1, R1, color='r', text='Cam1')
    plot_3d_camera(ax, cam2, R2, color='g', text='Cam2')


if __name__ == '__main__':
    path_3d = 'data/Point3D.txt'
    points_3d = read_3d_points(path_3d)
    num_points = points_3d.shape[0]
    points_3d = np.c_[points_3d, np.ones((num_points, 1))]
    P3 = points_3d.T
    camera_matrix = {}

    img = cv2.imread('data/chessboard_2.jpg')
    cv2.imshow("aaa", img)

    gray = img[..., 1]
    cv2.imshow("bbb", gray)

    count = np.zeros(256)
    print(gray.shape[0] * gray.shape[1])
    for p in gray.ravel():
        count[p] += 1
    print(np.sum(count))
    x = [xx for xx in range(256)]
    plt.plot(x, count)
    plt.show()

    threshold = 127
    binary = np.zeros(gray.shape)
    binary[gray > threshold] = 255

    h, w = gray.shape
    cv2.imshow("ccc", binary)

    res_tmp = signal.convolve2d(binary, np.array([[-1, 1]]), 'valid')
    res_x = np.zeros(gray.shape)
    res_x[:, 1:] = np.abs(res_tmp)
    del res_tmp
    res_tmp = signal.convolve2d(binary, np.array([[-1], [1]]), 'valid')
    res_y = np.zeros(gray.shape)
    res_y[1:, :] = np.abs(res_tmp)
    del res_tmp
    res = np.clip(res_x + res_y, 0, 255).astype('uint8')

    res2 = cv2.Canny(gray, 50, 150)

    cv2.imwrite("ddd.png", res)
    cv2.imwrite("eee.png", res2)

    #cv2.imshow("ddd", res)
    cv2.waitKey(0)
