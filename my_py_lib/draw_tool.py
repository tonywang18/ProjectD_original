'''
一些复杂的绘制方法，opencv中没有提供现成的绘制方法
注意，绘制方法复杂，尽量使用 numba 或 numpy向量化 加速
'''

import cv2
import numpy as np
import numba

assert numba.__version__ >= '0.48.0'


@numba.njit
def _njit_draw_gradient_circle(im: np.ndarray, center_yx: np.ndarray, radius: float, center_color, edge_color, mode=0):
    '''
    draw_gradient_circle numba加速
    :param im:
    :param center_yx:
    :param radius:
    :param center_color:
    :param edge_color:
    :return:
    '''
    yx_start = np.asarray(np.floor(center_yx - radius), np.int32)
    yx_end = np.asarray(np.ceil(center_yx + radius), np.int32)

    for y in range(yx_start[0], yx_end[0]):
        for x in range(yx_start[1], yx_end[1]):
            if 0 <= y < im.shape[0] and 0 <= x < im.shape[1]:
                v = np.asarray([center_yx[0] - y, center_yx[1] - x], np.float32)
                vlen = np.linalg.norm(v)
                if vlen <= radius:
                    # factor = func(vlen / radius)
                    if mode == 1:
                        factor = np.sqrt(vlen / radius)
                    elif mode == 2:
                        factor = np.square(vlen / radius)
                    else:
                        factor = vlen / radius
                    color = edge_color * factor + center_color * (1-factor)
                    im[y, x] = color
    return im


def draw_gradient_circle(im: np.ndarray, center_yx, radius, center_color, edge_color, mode='linear'):
    '''
    绘制一个颜色从中心到外围渐变的圆
    :param im:              输入图像
    :param center_yx:       圆心
    :param radius:          半径
    :param center_color:    中心颜色
    :param edge_color:      边缘颜色
    :return:
    '''
    import typing
    assert radius > 0, 'radius must be bigger than 0'
    assert isinstance(im, np.ndarray), 'im must be a numpy.ndarray'
    assert issubclass(im.dtype.type, typing.SupportsFloat), 'im must be a number array'
    assert im.ndim in (2, 3), 'only support ndarray ndim is 2 or 3'
    mode = ('linear', 'sqrt', 'square').index(mode)
    center_yx = np.asarray(center_yx, np.int32)
    center_color = np.asarray(center_color, im.dtype)
    edge_color = np.asarray(edge_color, im.dtype)
    _njit_draw_gradient_circle(im, center_yx, radius, center_color, edge_color, mode)
    return im


if __name__ == '__main__':
    import time

    # 测试 draw_gradient_circle
    a = np.zeros([512, 512, 3], np.uint8)
    t1 = time.time()
    draw_gradient_circle(a, [100, 100], 256, (255, 0, 0), (0, 255, 0))
    t2 = time.time()
    draw_gradient_circle(a, [100, 100], 256, (255, 0, 0), (0, 255, 0))
    t3 = time.time()
    print('draw_gradient_circle init time:', t2 - t1 - (t3 - t2))
    print('draw_gradient_circle exec time:', t3 - t2)
    cv2.imshow('a', a)
    cv2.waitKey(0)
