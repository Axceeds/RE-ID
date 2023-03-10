import numpy as np
import cv2 as cv


# 此文件中存放的方法是针对温度矩阵进行处理的函数，运算均针对单帧，与时序无关
def quantify(matrix, _min, resolution):
    # 对浮点数矩阵量化，将浮点数映射为无符号八位整数
    below = matrix < _min
    _max = _min + resolution * 256
    over = matrix >= _max
    normal = 1 - below - over
    new_matrix = ((matrix - _min) / resolution * normal + over * 255 + below * 0).astype('uint8')
    return new_matrix


def interpolate(matrix, points=0, new_size=(0, 0)):
    # 对矩阵进行插值 放大倍数points（浮点数）新矩阵大小newSize（元组，浮点数）
    if matrix.dtype != 'uint8':
        print('警告，矩阵数据格式不为uint8，不进行插值，返回原数据')
        return matrix
    if points <= 0:
        if new_size == (0, 0):
            return matrix
        else:
            return cv.resize(matrix, new_size, None, 0, 0)
    else:
        return cv.resize(matrix, (0, 0), None, points, points)


def spatial_filter(matrix, mode, win_size):
    # 对矩阵进行滤波窗口大小winSize（整数，奇数）
    if matrix.dtype != 'uint8':
        print('警告，矩阵数据格式不为uint8，不进行滤波，返回原数据')
        return matrix
    if mode == 'None':
        return matrix
    elif mode == 'average':
        win_size = int(win_size)
        return cv.blur(matrix, (win_size, win_size))
    elif mode == 'Gauss':
        win_size = int(win_size)
        return cv.GaussianBlur(matrix, (win_size, win_size), 0)
    elif mode == 'median':
        win_size = int(win_size)
        return cv.medianBlur(matrix, win_size)
    elif mode == 'bilateral':
        win_size = int(win_size)
        return cv.bilateralFilter(matrix, win_size, 45, 0)
    else:
        print('不存在的空间滤波方式，返回原数据')
        return matrix


def segment(matrix, mode, threshold=15):
    # 对矩阵进行分割，手动阈值threshold（浮点数）为了保持图像的显示效果，将减去一个最小值再进行阈值分割
    if matrix.dtype != 'uint8':
        print('警告，矩阵数据格式不为uint8，不进行阈值分割，返回原数据')
        return matrix
    new_matrix = matrix.copy()
    # 根据模式对矩阵进行阈值分割
    if mode == 'None':
        return new_matrix
    elif mode == 'artificial':
        binary = matrix > threshold
        return (new_matrix * binary).astype('uint8')
    elif mode == 'Otsu':
        return otsu_threshold_segmentation(new_matrix)
    elif mode == 'triangle':
        threshold, temp = cv.threshold(new_matrix, 0, 255, cv.THRESH_TOZERO + cv.THRESH_TRIANGLE)
        return temp
    elif mode == 'triangle2':
        return adaptive_threshold_segmentation(new_matrix)
    else:
        print('不存在的阈值分割方式，返回原数据')
        return new_matrix


def otsu_threshold_segmentation(matrix, binary_para=1):
    # 利用大津阈值对矩阵进行分割
    # 步长越小，找到的阈值就越精细。由于已经进行了量化，步长最小为1
    _min = float(matrix.min())
    _max = float(matrix.max())
    # 计算大津阈值
    max_g = 0
    suitable_th = 0
    st = binary_para
    layer = int((_max - _min) / (st)) + 1
    for threshold in range(round(layer)):
        bin_img = matrix > threshold * st + _min
        bin_img_inv = matrix <= threshold * st + _min
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue
        w0 = float(fore_pix) / matrix.size
        u0 = float(np.sum(matrix * bin_img)) / fore_pix
        w1 = float(back_pix) / matrix.size
        u1 = float(np.sum(matrix * bin_img_inv)) / back_pix
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = threshold
    threshold = suitable_th * st + _min
    binary = (matrix > threshold).astype('uint8')
    return (matrix * binary).astype('uint8')



def adaptive_threshold_segmentation(matrix):
    # 自适应阈值
    _hist = [0] * 256
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            _hist[matrix[i, j]] += 1

    _max = max(_hist)
    peak = (_hist.index(_max), _max)
    down = (255, 0)
    for i in range(peak[0], 255):
        if not _hist[i]:
            if i + 10 < 256:
                down = (i + 10, 0)
            else:
                down = (255, 0)
            break

    if peak[0] == down[0]:
        a = 1
        b = 0
        c = 0
    else:
        a = 1 / (down[0] - peak[0])
        b = 1 / peak[1]
        c = -1

    max_d = 0
    threshold = 0
    for i in range(peak[0], down[0] + 1):
        d = abs(a * (i - peak[0]) + b * _hist[i] + c) / ((a ** 2 + b ** 2) ** 0.5)
        if d > max_d:
            max_d = d
            threshold = i
    new_matrix = (matrix > threshold) * matrix
    return new_matrix.astype('uint8')

def connected_component(matrix, size):
    # 获取有效连通域(大于size的)
    # 有效连通域最小值size（整数）
    valid = np.zeros_like(matrix, dtype='bool')
    _, connected_component = cv.connectedComponents(matrix)  # 连通域分析，不同的连通域会被标记成1，2，3；背景为0
    for i in range(1, connected_component.max() + 1):
        area = (connected_component == i)
        if area.sum() >= size:
            valid += area
    return matrix * valid

def max_connected_component(matrix):
    # 获取最大连通域
    valid = np.zeros_like(matrix, dtype='bool')
    _, connected_component = cv.connectedComponents(matrix)  # 连通域分析，不同的连通域会被标记成1，2，3；背景为0
    max = 0
    for i in range(1, connected_component.max() + 1):
        area = (connected_component == i)
        if area.sum() > max:
            max = area.sum()
            valid = area
    return matrix * valid


def feature(matrix, gain=1):
    # 计算矩阵的特征值 伸缩因子gain（数值）
    # 输出：
    #     1、重心（x轴，y轴）
    #     2、加权标准差（x轴，y轴）
    # y轴和直觉的方向相反，需要注意
    row = matrix.shape[0]
    col = matrix.shape[1]
    if abs(row) < 1e-5 or abs(col) < 1e-5:
        return [0] * 6
    # 计算重心
    gx = 0
    gy = 0
    for i in range(row):
        for j in range(col):
            gy += i * matrix[i, j]
            gx += j * matrix[i, j]
    _sum = matrix.sum()
    gy /= _sum + 1
    gx /= _sum + 1

    # 计算加权标准差
    wvx = 0
    wvy = 0
    for i in range(row):
        for j in range(col):
            wvy += ((i - gy) ** 2) * matrix[i, j]
            wvx += ((j - gx) ** 2) * matrix[i, j]
    wvy /= _sum + 1
    wvx /= _sum + 1
    wvy = wvy ** 0.5
    wvx = wvx ** 0.5
    p = wvx / wvy if wvy >= 1 else wvy

    return gx, gy, wvx * gain, wvy * gain, p

def center(matrix):
    # 计算矩阵的重心
    # 输出：重心（x轴，y轴）
    # y轴和直觉的方向相反，需要注意
    row = matrix.shape[0]
    col = matrix.shape[1]

    # 计算重心
    gx = 0
    gy = 0
    for i in range(row):
        for j in range(col):
            gy += i * matrix[i, j]
            gx += j * matrix[i, j]
    _sum = matrix.sum()
    gy /= _sum + 1
    gx /= _sum + 1
    return gx, gy


def vector_to_matrix(array1, array2):
    # 计算两个向量相乘得到的矩阵
    array_len1 = len(array1)
    array_len2 = len(array2)
    if array_len1 == 0 or array_len2 == 0:
        return 0
    else:
        new_matrix = np.zeros((array_len1, array_len2))
        for i in range(array_len1):
            for j in range(array_len2):
                new_matrix[i, j] = array1[i] * array2[j]
        return new_matrix


def diff(matrix):
    """#######################################
    获取三帧差分
    输入：
        1、原始矩阵matrix（ndarray三维，时间维度长度为3）
    输出：
        1、三帧差分
    备注：
    #######################################"""
    diff1 = matrix[-1].astype('float64') - matrix[-2].astype('float64')
    diff2 = matrix[-2].astype('float64') - matrix[-3].astype('float64')
    diff1 = np.abs(diff1)
    diff2 = np.abs(diff2)
    diff = diff1 * diff2
    return diff


def reverse(matrix):
    # 水平翻转
    row = matrix.shape[0]
    col = matrix.shape[1]
    new_matrix = np.zeros_like(matrix)
    for i in range(row):
        for j in range(col):
            new_matrix[i, j, ...] = matrix[i, col - 1 - j, ...]
    return new_matrix


def move(matrix, pixels):
    # 图像左右平移
    # 输入：
    #     1、原始矩阵matrix（ndarray）
    #     2、平移像素个数pixels（整数）
    # 备注：
    #     1、左移为负，右移为正
    col = matrix.shape[1]
    new_matrix = np.zeros_like(matrix)
    if pixels > 0:
        new_matrix[:, :pixels] = matrix[:, col - pixels:]
        new_matrix[:, pixels:] = matrix[:, :col - pixels]
    else:
        new_matrix[:, col + pixels:] = matrix[:, :-pixels]
        new_matrix[:, :col + pixels] = matrix[:, -pixels:]
    return new_matrix

def stretch(matrix, pixels):
    """#######################################
    图像拉伸，然后去掉多余部分
    输入：
        1、拉伸像素数
    输出：
        1、处理后的数据
    备注：
    #######################################"""
    row = matrix.shape[0]
    col = matrix.shape[1]
    new_matrix = interpolate(matrix, new_size=(col + pixels, row + pixels))
    new_matrix = new_matrix[pixels:, pixels // 2:pixels // 2 + col]
    return new_matrix

def shrink(matrix, pixels):
    # 图像收缩，然后用最小值填充
    # 输入：
    #     1、拉伸像素数
    row = matrix.shape[0]
    col = matrix.shape[1]
    shrinked_matrix = interpolate(matrix, new_size=(col - pixels, row - pixels))
    new_matrix = np.ones_like(matrix) * shrinked_matrix.min() + np.random.randint(0, 5, matrix.shape)
    new_matrix[pixels:, pixels // 2:col - pixels // 2] = shrinked_matrix
    return new_matrix.astype('uint8')
