import numpy as np
import cv2
import os
import time

'''
Title: Homography Transformation
'''


def draw_rectange_line(imgA, imgB, imgC, outdir=''):
    #read image A
    points_A1 = np.load('img_A_1_points.npy').astype(np.int)
    points_A2 = np.load('img_A_2_points.npy').astype(np.int)

    n = points_A1.shape[0]
    x_min = int(np.min(points_A1[:, 0]))
    x_max = int(np.max(points_A1[:, 0]))
    y_min = int(np.min(points_A1[:, 1]))
    y_max = int(np.max(points_A1[:, 1]))
    maskA1_a = np.zeros([y_max-y_min+1, x_max-x_min+1], dtype=np.float)
    mstartA1_x = x_min
    mstartA1_y = y_min
    x_min = int(np.min(points_A2[:, 0]))
    x_max = int(np.max(points_A2[:, 0]))
    y_min = int(np.min(points_A2[:, 1]))
    y_max = int(np.max(points_A2[:, 1]))
    maskA2_a = np.zeros([y_max-y_min+1, x_max-x_min+1], dtype=np.float)
    mstartA2_x = x_min
    mstartA2_y = y_min
    for points, color, mask, start in \
            zip([points_A1, points_A2], [(0, 255, 255), (255, 0, 255)], [maskA1_a, maskA2_a], [np.array([mstartA1_x, mstartA1_y]), np.array([mstartA2_x, mstartA2_y])]):
        cv2.line(imgA, tuple(points[0]), tuple(points[1]), color, 3)
        cv2.line(imgA, tuple(points[1]), tuple(points[3]), color, 3)
        cv2.line(imgA, tuple(points[0]), tuple(points[2]), color, 3)
        cv2.line(imgA, tuple(points[2]), tuple(points[3]), color, 3)
        cv2.line(mask, tuple(points[0]-start), tuple(points[1]-start), 255, 1)
        cv2.line(mask, tuple(points[1]-start), tuple(points[3]-start), 255, 1)
        cv2.line(mask, tuple(points[0]-start), tuple(points[2]-start), 255, 1)
        cv2.line(mask, tuple(points[2]-start), tuple(points[3]-start), 255, 1)

    maskA1_a = maskA1_a == 255
    maskA2_a = maskA2_a == 255
    maskA1 = (maskA1_a, mstartA1_x, mstartA1_y)
    maskA2 = (maskA2_a, mstartA2_x, mstartA2_y)
    cv2.imwrite(os.path.join(outdir, "A.png"), imgA)

    points_B = np.load('img_B_points.npy').astype(np.int)
    points_C = np.load('img_C_points.npy').astype(np.int)
    x_min = int(np.min(points_B[:, 0]))
    x_max = int(np.max(points_B[:, 0]))
    y_min = int(np.min(points_B[:, 1]))
    y_max = int(np.max(points_B[:, 1]))
    maskB_a = np.zeros([y_max - y_min + 1, x_max - x_min + 1], dtype=np.float)
    mstartB_x = x_min
    mstartB_y = y_min
    x_min = int(np.min(points_C[:, 0]))
    x_max = int(np.max(points_C[:, 0]))
    y_min = int(np.min(points_C[:, 1]))
    y_max = int(np.max(points_C[:, 1]))
    maskC_a = np.zeros([y_max - y_min + 1, x_max - x_min + 1], dtype=np.float)
    mstartC_x = x_min
    mstartC_y = y_min

    for points, img, mask, start in\
            zip([points_B, points_C], [imgB, imgC], [maskB_a, maskC_a], [np.array([mstartB_x, mstartB_y]), np.array([mstartC_x, mstartC_y])]):
        cv2.line(img, tuple(points[0]), tuple(points[1]), (0, 255, 255), 3)
        cv2.line(img, tuple(points[1]), tuple(points[3]), (0, 255, 255), 3)
        cv2.line(img, tuple(points[0]), tuple(points[2]), (0, 255, 255), 3)
        cv2.line(img, tuple(points[2]), tuple(points[3]), (0, 255, 255), 3)
        cv2.line(mask, tuple(points[0]-start), tuple(points[1]-start), 255, 1)
        cv2.line(mask, tuple(points[1]-start), tuple(points[3]-start), 255, 1)
        cv2.line(mask, tuple(points[0]-start), tuple(points[2]-start), 255, 1)
        cv2.line(mask, tuple(points[2]-start), tuple(points[3]-start), 255, 1)

    maskB_a = maskB_a == 255
    maskC_a = maskC_a == 255
    maskB = (maskB_a, mstartB_x, mstartB_y)
    maskC = (maskC_a, mstartC_x, mstartC_y)

    cv2.imwrite(os.path.join(outdir, "B.png"), imgB)
    cv2.imwrite(os.path.join(outdir, "C.png"), imgC)

    del imgA, imgB, imgC, img
    del points_A1, points_A2, points_B, points_C, points
    return [maskA1, maskA2, maskB, maskC]


def getH(S, D):
    n = S.shape[0]
    A = np.zeros((2*n, 9), dtype=np.float64)
    h = 0
    for i in range(n):
        x, y = tuple(D[i, :])
        A[h, :2] = S[i, :]
        A[h, 2] = 1
        A[h+1, 3:5] = S[i, :]
        A[h+1, 5] = 1
        A[h, 6:8] = -x * S[i, :]
        A[h, 8] = -x
        A[h+1, 6:8] = -y * S[i, :]
        A[h+1, 8] = -y
        h += 2

    U, _, V = np.linalg.svd(A)
    P = V.T[:, 8]
    return P.reshape((3, 3))


def forwarding_nearest(dst_p, imgsrc, imgdst, weight, x, y):
    dst_x, dst_y = np.int(np.round(dst_p[0])), np.int(np.round(dst_p[1]))
    h, w = imgdst.shape[:2]
    if dst_y > h-1:
        dst_y = h-1
    if dst_x > w-1:
        dst_x = w-1
    if weight[dst_y, dst_x] == 0:
        imgdst[dst_y, dst_x] = np.array([0, 0, 0])
    imgdst[dst_y, dst_x] += imgsrc[y, x]
    weight[dst_y, dst_x] += 1


def forwarding_bilinear(dst_p, imgsrc, imgdst, weight, x, y):
    x_floor, y_floor = np.int(np.floor(dst_p[0])), np.int(np.floor(dst_p[1]))
    h, w = imgdst.shape[:2]
    x_rate = dst_p[0] - x_floor
    y_rate = dst_p[1] - y_floor
    x_next = x_floor if x_floor > w-2 else x_floor + 1
    y_next = y_floor if y_floor > h - 2 else y_floor + 1
    for dst_x, xrate in zip([x_floor, x_next], [1-x_rate, x_rate]):
        for dst_y, yrate in zip([y_floor, y_next], [1 - y_rate, y_rate]):
            if weight[dst_y, dst_x] == 0:
                imgdst[dst_y, dst_x] = np.array([0, 0, 0])
            weight[dst_y, dst_x] += xrate * yrate
            imgdst[dst_y, dst_x] += xrate * yrate * imgsrc[y, x]


def do_forward_projection(H, imgsrc, srcmask, imgdst, dstmask, method='bilinear'):
    srcmask, srcstart_x, srcstart_y = srcmask
    dstmask, dststart_x, dststart_y = dstmask
    start_x = srcmask.argmax(axis=1) + srcstart_x
    y_range = np.nonzero(start_x)[0] + srcstart_y
    start_y, end_y = y_range[0], y_range[-1]

    if method == 'nearest':
        mapping_func = forwarding_nearest
    else:
        mapping_func = forwarding_bilinear

    weight = np.zeros(imgdst.shape[:2], dtype=np.float64)
    imgdst = imgdst.astype(np.float64)
    print('Projecting:')
    for y in range(start_y+1, end_y):
        print("\r%3.1f%%" % ((y-start_y) / (end_y-start_y) * 100), end='')
        x = start_x[y-srcstart_y]
        border = np.argwhere(srcmask[y-srcstart_y]).flatten()
        try:
            while srcmask[y-srcstart_y, x-srcstart_x]:
                x += 1
        except IndexError:
            pass
        if border[-1] == x-srcstart_x-1:
            continue
        while not srcmask[y-srcstart_y, x-srcstart_x]:
            dst_p = H @ np.array([x, y, 1])
            dst_p /= dst_p[2]
            mapping_func(dst_p, imgsrc, imgdst, weight, x, y)
            x += 1
    print("\r100%           ")

    ys, xs = np.nonzero(weight)
    for i in range(xs.shape[0]):
        if weight[ys[i], xs[i]] != 0:
            imgdst[ys[i], xs[i]] = imgdst[ys[i], xs[i]] / weight[ys[i], xs[i]]

    start_x_d = dstmask.argmax(axis=1) + dststart_x
    y_range_d = np.nonzero(start_x_d)[0] + dststart_y
    start_y_d, end_y_d = y_range_d[0], y_range_d[-1]

    print('Fixing:')
    for y in range(start_y_d+1, end_y_d):
        print("\r%3.1f%%       " % ((y-start_y_d) / (end_y_d-start_y_d) * 100), end='')
        x = start_x_d[y-dststart_y]
        border = np.argwhere(dstmask[y-dststart_y]).flatten()
        try:
            while dstmask[y-dststart_y, x-dststart_x]:
                x += 1
        except IndexError:
            pass
        if border[-1] == x-dststart_x-1:
            continue
        while not dstmask[y-dststart_y, x-dststart_x]:
            if weight[y, x] == 0:
                right = 1
                while weight[y, x+right] == 0 and not dstmask[y-dststart_y, x-dststart_x+right]:
                    right += 1
                down = 1
                while weight[y+down, x] == 0 and not dstmask[y-dststart_y+down, x-dststart_x]:
                    down += 1
                lf_color = imgdst[y, x-1] * (right / (1+right)) + (1 / (1+right)) * imgdst[y, x+right]
                td_color = imgdst[y-1, x] * (down / (1+down)) + (1 / (1+down)) * imgdst[y+down, x]
                imgdst[y, x] = (lf_color + td_color) / 2
            x += 1
    print("\r100%              ")

    return np.uint8(np.round(imgdst))


def get_color_bilinear_inverse(res_s, imgsrc):
    x_floor, y_floor = np.int(np.floor(res_s[0])), np.int(np.floor(res_s[1]))
    h, w = imgsrc.shape[:2]
    x_rate = res_s[0] - x_floor
    y_rate = res_s[1] - y_floor
    x_next = x_floor if x_floor > w-2 else x_floor + 1
    y_next = y_floor if y_floor > h - 2 else y_floor + 1
    return x_rate * y_rate * imgsrc[y_next, x_next] + \
           (1 - x_rate) * y_rate * imgsrc[y_next, x_floor] + \
           x_rate * (1 - y_rate) * imgsrc[y_floor, x_next] + \
           (1 - x_rate) * (1 - y_rate) * imgsrc[y_floor, x_floor]


def get_color_nearest_inverse(res_s, imgsrc):
    x_cap, y_cap = np.int(np.round(res_s[0])), np.int(np.round(res_s[1]))
    h, w = imgsrc.shape[:2]
    if x_cap > w-1:
        x_cap = w - 1
    if y_cap > h - 1:
        y_cap = h-1
    return imgsrc[y_cap, x_cap]


def do_inverse_projection(H, imgsrc, srcmask, imgdst, dstmask, method='bilinear'):
    srcmask, srcstart_x, srcstart_y = srcmask
    dstmask, dststart_x, dststart_y = dstmask
    start_x_d = dstmask.argmax(axis=1) +dststart_x
    y_range_d = np.nonzero(start_x_d)[0] + dststart_y
    start_y_d, end_y_d = y_range_d[0], y_range_d[-1]

    if method == 'nearest':
        get_color = get_color_nearest_inverse
    else:
        get_color = get_color_bilinear_inverse

    print('Projection:')
    for y in range(start_y_d + 1, end_y_d):
        print("\r%3.1f%%          " % ((y - start_y_d) / (end_y_d - start_y_d) * 100), end='')
        x = start_x_d[y-dststart_y]
        border = np.argwhere(dstmask[y-dststart_y]).flatten()
        try:
            while dstmask[y-dststart_y, x-dststart_x]:
                x += 1
        except IndexError:
            pass
        if border[-1] == x-dststart_x - 1:
            continue
        while not dstmask[y-dststart_y, x-dststart_x]:
            res_s = H @ np.array([x, y, 1])
            res_s = res_s / res_s[2]
            imgdst[y, x] = get_color(res_s, imgsrc)
            x += 1
    print("\r100%                ")

    return np.uint8(np.round(imgdst))


if __name__ == '__main__':
    imgA = cv2.imread('photos/A.jpg')
    imgB = cv2.imread('photos/B.jpg')
    imgC = cv2.imread('photos/C.jpg')
    outdir = 'hw2_2_results/'

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    mask = draw_rectange_line(imgA.copy(), imgB.copy(), imgC.copy(), outdir)

    points_A1 = np.load('img_A_1_points.npy').astype(np.int)
    points_A2 = np.load('img_A_2_points.npy').astype(np.int)
    points_B = np.load('img_B_points.npy').astype(np.int)
    points_C = np.load('img_C_points.npy').astype(np.int)

    # cv2.imshow("dd", mask[0].astype('float'))
    
    # do image A forwarding projection
    t = time.time()
    print('Image A forward projection process')
    H = getH(points_A1, points_A2)
    imgA_dst = imgA.copy()
    imgA_dst = do_forward_projection(H, imgA, mask[0], imgA_dst, mask[1])
    H = getH(points_A2, points_A1)
    imgA_dst = do_forward_projection(H, imgA, mask[1], imgA_dst, mask[0])
    cv2.imwrite(os.path.join(outdir, "imgA_forward_result.png"), imgA_dst)
    del imgA_dst
    print(time.time()-t)

    # do image A inverse projection
    t = time.time()
    print('Image A backward projection process')
    H = getH(points_A2, points_A1)
    imgA_dst = imgA.copy()
    imgA_dst = do_inverse_projection(H, imgA, mask[0], imgA_dst, mask[1])
    H = getH(points_A1, points_A2)
    imgA_dst = do_inverse_projection(H, imgA, mask[1], imgA_dst, mask[0])
    cv2.imwrite(os.path.join(outdir, "imgA_backward_result.png"), imgA_dst)
    print(time.time() - t)
    
    # do image B C forwarding projection
    t = time.time()
    print('Image B, C forward projection process')
    H = getH(points_B, points_C)
    imgB_dst = imgB.copy()
    imgC_dst = imgC.copy()
    imgC_dst = do_forward_projection(H, imgB, mask[2], imgC_dst, mask[3])
    H = getH(points_C, points_B)
    imgB_dst = do_forward_projection(H, imgC, mask[3], imgB_dst, mask[2])

    cv2.imwrite(os.path.join(outdir, "imgB_forward_result.png"), imgB_dst)
    cv2.imwrite(os.path.join(outdir, "imgC_forward_result.png"), imgC_dst)
    del imgB_dst, imgC_dst
    print(time.time() - t)

    # do image B C inverse projection
    t = time.time()
    print('Image B, C backward projection process')
    H = getH(points_C, points_B)
    imgB_dst = imgB.copy()
    imgC_dst = imgC.copy()
    imgC_dst = do_inverse_projection(H, imgB, mask[2], imgC_dst, mask[3])
    H = getH(points_B, points_C)
    imgB_dst = do_inverse_projection(H, imgC, mask[3], imgB_dst, mask[2])

    cv2.imwrite(os.path.join(outdir, "imgB_backward_result.png"), imgB_dst)
    cv2.imwrite(os.path.join(outdir, "imgC_backward_result.png"), imgC_dst)
    print(time.time() - t)

    cv2.waitKey(-1)