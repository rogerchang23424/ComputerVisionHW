import numpy as np
import cv2
import sys
from scipy import signal
import math
import re
import matplotlib.pyplot as plt
import os

'''
Title: Harris Corner Detection
'''


def gaussian_smooth(img, ksize=5, sigma=5, padding='mirror', out_type=np.uint8):
    _ksize = np.int(ksize)
    _sigma = np.double(sigma)
    _img = img.copy()
    if len(_img.shape) == 3:
        h, w, depth = _img.shape
    elif len(_img.shape) == 2:
        h, w = _img.shape
        depth = 1
        _img.shape = (h, w, 1)

    window = np.zeros((ksize, ), dtype=np.float64)
    half = _ksize // 2

    is_odd = (_ksize % 2 == 1)
    window[half] = np.float64(1.)
    if is_odd:
        for i in range(1, int(half)+1):
            t = np.float64(i) / _sigma
            window[half+np.int(i)] = window[half-np.int(i)] = np.exp(-0.5 * t * t)
    else:
        for i in range(1, int(half)):
            t = np.float64(i) / _sigma
            window[half+np.int(i)] = window[half-np.int(i)] = np.exp(-0.5 * t * t)

        t = np.float64(half) / _sigma
        window[0] = np.exp(-0.5 * t * t)

    window = window / np.sum(window)

    print(window)

    _tmp = np.zeros((h+ksize-1, w+ksize-1, depth), dtype=np.float64)
    _tmp[half:h+half, half:w+half] = _img[:, :]

    if padding == 'mirror':
        if is_odd:
            for i in range(half):
                _i = i+1
                _tmp[half-_i, half-i:w+half+i] = _tmp[half+i, half-i:w+half+i]
                _tmp[half-i:h+half+i, half-_i] = _tmp[half-i:h+half+i, half+i]
                _tmp[h+half+i, half-i:w+half+i] =_tmp[h+half-_i, half-i:w+half+i]
                _tmp[half-i:h+half+i, w+half+i] = _tmp[half-i:h+half+i, w+half-_i]

                _tmp[half-_i, half-_i] = _tmp[half+i, half+i]
                _tmp[half-_i, w+half+i] = _tmp[half+i, w+half-_i]
                _tmp[h+half+i, half-_i] = _tmp[h+half-_i, half+i]
                _tmp[h+half+i, w+half+i] = _tmp[h+half-_i, w+half-_i]
        else:
            for i in range(half-1):
                _i = i+1
                _tmp[half-_i, half-i:w+half+i] = _tmp[half+i, half-i:w+half+i]
                _tmp[half-i:h+half+i, half-_i] = _tmp[half-i:h+half+i, half+i]
                _tmp[h+half+i, half-i:w+half+i] =_tmp[h+half-_i, half-i:w+half+i]
                _tmp[half-i:h+half+i, w+half+i] = _tmp[half-i:h+half+i, w+half-_i]

                _tmp[half-_i, half-_i] = _tmp[half+i, half+i]
                _tmp[half-_i, w+half+i] = _tmp[half+i, w+half-_i]
                _tmp[h+half+i, half-_i] = _tmp[h+half-_i, half+i]
                _tmp[h+half+i, w+half+i] = _tmp[h+half-_i, w+half-_i]

            _tmp[0, 1:w+half*2-1] = _tmp[half*2-1, 1:w+half*2-1]
            _tmp[1:h+half*2-1, 0] = _tmp[1:h+half*2-1, half*2-1]
            _tmp[0, 0] = _tmp[half*2-1, half*2-1]
    elif padding == 'zero':
        pass
    else:
        raise Exception('')

    out1 = np.zeros((h+ksize-1, w, depth), dtype=np.float64)
    window = window.reshape(1, ksize)
    for d in range(depth):
        out_t = signal.convolve2d(_tmp[:, :, d], window, 'valid')
        out1[:, :, d] = out_t
        del out_t
    del _tmp

    res = np.zeros((h, w, depth), dtype=out_type)
    window = window.reshape(ksize, 1)
    for d in range(depth):
        out_t = signal.convolve2d(out1[:, :, d], window, 'valid')
        if not (out_type == np.float64 or out_type == np.float32):
            res[:, :, d] = out_type(np.round(out_t))
        else:
            res[:, :, d] = out_type(out_t)
        del out_t
    del out1

    return res


def sobel_edge_detection(img, threshold=20, colormap='viridis', skip_colormap=False):
    h, w = img.shape

    hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

    _tmp = np.ndarray((h+2, w+2), dtype=np.uint8)
    _tmp[1:h+1, 1:w+1] = img[:, :]

    if True:
        _tmp[0, 1:w+1] = _tmp[1, 1:w+1]
        _tmp[1:h+1, 0] = _tmp[1:h+1, 1]
        _tmp[h+1, 1:w+1] = _tmp[h, 1:w+1]
        _tmp[1:h+1, w+1] = _tmp[1:h+1, w]

        _tmp[0, 0] = _tmp[1, 1]
        _tmp[0, w+1] = _tmp[1, w]
        _tmp[h+1, 0] = _tmp[h, 1]
        _tmp[h+1, w+1] = _tmp[h, w]

    res_x = signal.convolve2d(_tmp, hx, 'valid')
    res_y = signal.convolve2d(_tmp, hy, 'valid')

    t = np.sqrt(np.square(res_x)+np.square(res_y))
    mag = np.uint8(np.minimum(np.round(t), 255))
    mag_c = mag.copy()
    mag_c[mag<=threshold] = 0

    gra = np.arctan(res_y / res_x)

    h_pi = np.pi / 2

    gra[np.isnan(gra)] = h_pi
    gra[mag <= threshold] = - 2.
    gra_colormap = np.ndarray((h, w, 3), dtype=np.uint8)

    if not skip_colormap:
        print('Generating Color Map')
        colors = plt.get_cmap(colormap).colors
        for i in range(h):
            print('\r%3.2f%%' % (i/h*100), end='')
            for j in range(w):
                if gra[i, j] < - h_pi:
                    gra_colormap[i, j] = np.array([0, 0, 0], dtype=np.uint8)
                else:
                    s = np.round((gra[i, j] + h_pi) * 255 / np.pi)
                    gra_colormap[i, j] = np.uint8(np.round(np.dot(mag_c[i, j], colors[np.int(s)])))
        print('\r100%        ')

    res_x[mag<=threshold] = 0
    res_y[mag<=threshold] = 0
    return mag_c, gra_colormap, res_x, res_y


def structure_tensor(img_x, img_y, ksize=3, threshold=0.6):
    Ixx = np.square(img_x)
    Iyy = np.square(img_y)
    Ixy = img_x * img_y

    h, w = img_x.shape
    half = ksize // 2
    A = gaussian_smooth(Ixx, ksize, sigma=2*half/3, out_type=np.float64)
    B = gaussian_smooth(Ixy, ksize, sigma=2*half/3, out_type=np.float64)
    C = gaussian_smooth(Iyy, ksize, sigma=2*half/3, out_type=np.float64)
    A.shape = (h, w)
    B.shape = (h, w)
    C.shape = (h, w)

    return A, B, C


def compute_corner_strength(A, B, C, k=0.04):
    return (A*C-np.square(B)) - k * np.square(A+C)


def nms(R, threshold, radius=5):
    h, w = R.shape
    skip = (R < threshold)
    corner_img = np.zeros((h, w), dtype=np.uint8)
    corners = []

    ssize = h-2*radius
    for i in range(radius, h-radius):
        print('\r%3.1f%%' % ((i-radius)*100 / ssize), end='')
        j = radius
        while j < w-radius-1 and (skip[i, j] or R[i, j-1] >= R[i, j]):
            j += 1

        while j < w-radius-1:
            while j < w-radius-1 and (skip[i, j] or R[i, j+1] >= R[i, j]):
                j += 1

            if j < w-radius-1:
                p_1 = j + 2
                while p_1 <= j + radius and R[i, p_1] < R[i, j]:
                    skip[i, p_1] = True
                    p_1 += 1

                if p_1 > j + radius:
                    p_2 = j-1

                    while p_2 >= j - radius and R[i, p_2] <= R[i, j]:
                        p_2 -= 1

                    if p_2 < j - radius:
                        k = i + radius
                        found = False

                        while not found and k > i:
                            l = j+radius
                            while not found and l >= j - radius:
                                if R[k,l] > R[i,j]:
                                    found = True
                                else:
                                    skip[k, l] = True
                                l = l - 1
                            k = k-1

                        k = i - radius
                        while not found and k < i:
                            l = j - radius
                            while not found and l <= j + radius:
                                if R[k,l] > R[i,j]:
                                    found = True
                                l += 1
                            k += 1

                        if not found:
                            corner_img[i, j] = R[i, j]
                            corners.append((i, j))
                j = p_1

    print('\r100%         ')
    return corner_img, corners


def rotate30(img):
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, 30, 1.0)
    h, w = img.shape[:2]
    res = cv2.warpAffine(img, rot_mat, (w, h))
    return res


def get_ops(args):
    paths = []
    options = {}
    i = 0
    while i < len(args):
        if '--' == args[i][:2]:
            if '=' in args[i]:
                tmp = args[i].split('=')
                options[tmp[0][2:]] = tmp[1]
            else:
                options[args[i][2:]] = ''
        else:
            if args[i] != '':
                paths.append(args[i])
        i += 1
    return paths, options


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: hw1_1.py [options] <file_path>', file=sys.stderr)
        sys.exit(-1)

    paths, options = get_ops(sys.argv[1:])

    dot_color_name = options.get('dot_color', 'red')
    if dot_color_name == 'green':
        dot_color = [0, 255, 0]
    elif dot_color_name == 'blue':
        dot_color = [0, 0, 255]
    else:
        dot_color = [255, 0, 0]

    dot_radius = np.int(options.get('dot_radius', 2))

    sobel_edge_threshold = np.int(options.get('sobel_edge_threshold', 20))

    colormap = options.get('sobel_edge_colormap', 'viridis')

    R_k = float(options.get('R_k', '0.04'))

    nms_threshold = options.get('nms_threshold', None)
    if nms_threshold:
        nms_threshold = int(nms_threshold)

    nms_threshold_rate = float(options.get('nms_threshold_rate', 0.01))

    path = paths[0]
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print('Error: %s does not exist!' % path, file=sys.stderr)
        sys.exit(-1)

    m = re.match(r'.+?[\\/]([^\\/]+)\.[^\.]+', path)
    if m:
        filename = m.group(1)
    else:
        filename = path

    out_dir = 'results/'
    src = cv2.imread(path)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    g_res_5 = gaussian_smooth(src, 5, 5)

    # t = cv2.imread(out_dir+'%s_gaussian_k=5_2.png' % filename)
    # t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
    # print(psnr(t, g_res_5))
    # del t
 
    cv2.imwrite(out_dir+'%s_gaussian_k=5.png' % filename, cv2.cvtColor(g_res_5, cv2.COLOR_RGB2BGR))

    g_res_10 = gaussian_smooth(src, 10, 5)
    
    # t = cv2.imread(out_dir+'%s_gaussian_k=10_2.png' % filename)
    # t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
    # print(psnr(t, g_res_10))
    
    cv2.imwrite(out_dir+'%s_gaussian_k=10.png' % filename, cv2.cvtColor(g_res_10, cv2.COLOR_RGB2BGR))

    gray_res_5 = cv2.cvtColor(g_res_5, cv2.COLOR_RGB2GRAY)
    mag_5, gra_5, img_x5, img_y5 = sobel_edge_detection(gray_res_5, threshold=sobel_edge_threshold, colormap=colormap)
    cv2.imwrite(out_dir+'%s_magnitude_k=5.png' % filename, cv2.cvtColor(mag_5, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_dir+'%s_direction_k=5.png' % filename, cv2.cvtColor(gra_5, cv2.COLOR_RGB2BGR))

    gray_res_10 = cv2.cvtColor(g_res_10, cv2.COLOR_RGB2GRAY)
    mag_10, gra_10, img_x10, img_y10 = sobel_edge_detection(gray_res_10, threshold=sobel_edge_threshold, colormap=colormap)
    cv2.imwrite(out_dir + '%s_magnitude_k=10.png' % filename, cv2.cvtColor(mag_10, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_dir + '%s_direction_k=10.png' % filename, cv2.cvtColor(gra_10, cv2.COLOR_RGB2BGR))

    for window_size in [3, 30]:
        radius = 4
        a3, b3, c3 = structure_tensor(img_x5.copy(), img_y5.copy(), window_size)
        R = compute_corner_strength(a3, b3, c3, R_k)
        if nms_threshold is None:
            _threshold = nms_threshold_rate * R.max()
        else:
            _threshold = nms_threshold
        img2, corners = nms(R, _threshold, radius)

        del img2
        img = src.copy()
        print('Corners: %d' % len(corners))
        for corner in corners:
            i, j = corner
            cv2.circle(img, (j, i), dot_radius, dot_color, -1)

        # img_t = src.copy()
        # img_t[R>=_threshold] = [255, 0, 0]

        # cv2.imwrite(out_dir + '%s_corners_without_nms_w=%d.png' % (filename, window_size), cv2.cvtColor(img_t, cv2.COLOR_RGB2BGR))
        cv2.imwrite(out_dir + '%s_corners_w=%d.png' % (filename, window_size), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        del img

    img = src.copy()
    img2 = rotate30(img)
    del img
    img = src.copy()
    h ,w = img.shape[:2]
    img3 = cv2.resize(img, (w//2, h // 2))
    del img

    out_str = ['rotate30', 'scale0.5']
    for ii, img in enumerate([img2, img3]):
        g_res = gaussian_smooth(img, 5, 5)
        gray_res = cv2.cvtColor(g_res, cv2.COLOR_RGB2GRAY)
        _, _, img_x5, img_y5 = sobel_edge_detection(gray_res, threshold=sobel_edge_threshold, skip_colormap=True, colormap=colormap)

        window_size = 3
        radius = 4
        A, B, C = structure_tensor(img_x5, img_y5, window_size)
        R = compute_corner_strength(A, B, C, R_k)
        if nms_threshold is None:
            _threshold = nms_threshold_rate * R.max()
        else:
            _threshold = nms_threshold
        _, corners = nms(R, _threshold, radius)
        del _

        img_t = img.copy()
        print('Corners: %d' % len(corners))
        for corner in corners:
            i, j = corner
            cv2.circle(img_t, (j, i), dot_radius, dot_color, -1)

        cv2.imwrite(out_dir + '%s_%s_corners.png' % (filename, out_str[ii]), cv2.cvtColor(img_t, cv2.COLOR_RGB2BGR))