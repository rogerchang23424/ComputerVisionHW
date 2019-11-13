import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.linalg

'''
Title: Camera Calibration
'''


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

    outdir = 'hw2_1_results/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for i in range(1, 3):
        points_2d = np.load('Point2D_%d.npy' % i)
        img = cv2.imread('data/chessboard_%d.jpg' % i)
        points_2d = np.c_[points_2d, np.ones((points_2d.shape[0], 1))]
        P2 = points_2d.T

        P = calibrate(points_2d, points_3d)
        print('The result of P is: %s' % str(P))

        M = P[:, :3]
        K, R = scipy.linalg.rq(M)
        tmp = K[2,2]
        K = K / K[2, 2]
        if K[0, 0] < 0:
            K[:, 0] = -1 * K[:, 0]
            R[0, :] = -1 * R[0, :]

        if K[1, 1] < 0:
            K[:, 1] = -1 * K[:, 1]
            R[1, :] = -1 * R[1, :]

        t = np.matmul(np.linalg.inv(K), P[:, 3] / tmp)

        camera_matrix['cam_%d' % i] = {"P": {'dtype': P.dtype.__str__(), 'array': P.tolist()},
                                       "K": {'dtype': K.dtype.__str__(), 'array': K.tolist()},
                                       'R': {'dtype': R.dtype.__str__(), 'array': R.tolist()},
                                       't': {'dtype': t.dtype.__str__(), 'array': t.tolist()}}

        Rt = np.c_[R, t]
        proj_2d = np.matmul(np.matmul(K, Rt), points_3d.T).T
        for index in range(points_3d.shape[0]):
            proj_2d[index, :] = proj_2d[index, :] / proj_2d[index, 2]

        for pindex in range(points_2d.shape[0]):
            x_cap, y_cap, _ = tuple(points_2d[pindex, :])
            x, y, _ = tuple(proj_2d[pindex, :])

            cv2.circle(img, (np.int(np.round(x)), np.int(np.round(y))), 3, (0, 0, 255), -1)
            cv2.circle(img, (np.int(x_cap), np.int(y_cap)), 3, (0, 255, 255))

        error = (points_2d - proj_2d)[:, :2]
        rmse = np.sqrt(np.sum(np.square(error)) / points_2d.shape[0])

        print("RMSE is %f" % float(rmse))
        cv2.imwrite(os.path.join(outdir, "image_%d_RMSE=%f.png" % (i, rmse)), img)

    with open(os.path.join(outdir, 'camera_matrix.json'), 'w', encoding='utf-8') as f:
        json.dump(camera_matrix, f)

    R1 = np.array(camera_matrix['cam_1']['R']['array'], dtype=camera_matrix['cam_1']['R']['dtype'])
    t1 = np.array(camera_matrix['cam_1']['t']['array'], dtype=camera_matrix['cam_1']['t']['dtype'])
    R2 = np.array(camera_matrix['cam_2']['R']['array'], dtype=camera_matrix['cam_2']['R']['dtype'])
    t2 = np.array(camera_matrix['cam_2']['t']['array'], dtype=camera_matrix['cam_2']['t']['dtype'])

    C1 = -np.matmul(R1.T, t1)
    C2 = -np.matmul(R2.T, t2)

    
    #plot 3d image
    fig = plt.figure(figsize=(10,10), dpi=72)
    plot_3d_scene(fig, C1, R1, C2, R2)
    pos1 = R1.T[:, 2]
    pos1 = pos1 / np.linalg.norm(pos1)
    pos2 = R2.T[:, 2]
    pos2 = pos2 / np.linalg.norm(pos2)
    cos_d = np.dot(pos1, pos2)
    theta = np.arccos(cos_d)
    angle = theta / np.pi * 180
    print("Angle: %f" % angle)
    plt.show()
    
    cv2.waitKey(-1)
