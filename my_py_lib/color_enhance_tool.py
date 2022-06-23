'''
图像颜色增强工具
目前限定图像必须为RGB图像，uint8类型
'''

import cv2
import numpy as np
from typing import Iterable, Union


def _get_random_params(x_range: Union[int, float, Iterable], not_allow_negative_num=False):
    '''
    输入一个标量或一个区间，生成一个随机数
    :param x_range:
    :param not_allow_negative_num: 是否不允许返回负数
    :return:
    '''
    # 如果输入为0，直接返回
    if x_range == 0:
        return x_range
    # 如果为其他，则先构造一个最小最大区间，然后随机取一个数值
    if not isinstance(x_range, Iterable):
        x_range = [-x_range, x_range]
    if not_allow_negative_num:
        x_range = np.clip(x_range, 0, None)
    y = np.random.uniform(x_range[0], x_range[1])
    return y


def random_params_adjust_HSV(h_range=0., s_range=0., v_range=0.):
    '''
    随机参数用于图像的HSL空间进行调整
    :param h_range: 调整色调，值域[-1, 1]
    :param s_range: 调整对比度，值域[-1, 1]
    :param v_range: 调整亮度，值域[-1, 1]
    :return:
    '''
    hsv_add = np.array([_get_random_params(h_range),
                        _get_random_params(s_range),
                        _get_random_params(v_range)])
    return hsv_add


def adjust_HSV(img, h_add, s_add, v_add):
    '''
    在图像的HSL空间进行调整
    :param img: 输入的图像
    :param h_range: 调整色调，值域[-1, 1]
    :param s_range: 调整对比度，值域[-1, 1]
    :param v_range: 调整亮度，值域[-1, 1]
    :param return_params: 是否返回随机参数
    :param fix_params: 使用固定参数参数进行变换
    :return:
    '''
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    hsv = np.asarray(hsv, np.float32) / 255.

    hsv_add = [h_add, s_add, v_add]
    hsv = hsv + np.reshape(hsv_add, [1, 1, 3])
    # 注意色调是一个圆，可以循环的，所以使用求余
    # 令小于0.的H值+1得到正数，避免求余歧义
    hsv[:, :, 0] = np.where(hsv[:, :, 0] < 0, hsv[:, :, 0]+1, hsv[:, :, 0])
    hsv[:, :, 0] %= 1.
    hsv = np.clip(hsv, 0, 1.) * 255.
    hsv = hsv.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)

    return img


def random_adjust_HSV(img, h_range=0., s_range=0., v_range=0.):
    '''
    随机参数用于图像的HSL空间进行调整
    :param img: 输入图像
    :param h_range: 调整色调，值域[-1, 1]
    :param s_range: 调整对比度，值域[-1, 1]
    :param v_range: 调整亮度，值域[-1, 1]
    :return:
    '''
    params = random_params_adjust_HSV(h_range, s_range, v_range)
    img = adjust_HSV(img, *params)
    return img


def random_params_adjust_contrast_brightness(contrast_range: Union[float, Iterable]=0., brightness_range: Union[float, Iterable]=0.):
    '''
    获得对图像的对比度和亮度进行随机调整的参数
    :param img: 输入的图像
    :param contrast_range: 调整对比度，值域为(-1., +inf)
    :param brightness_range: 调整亮度，值域为(-1., +inf)
    :return:
    '''
    contrast = _get_random_params(contrast_range, not_allow_negative_num=True) + 1
    brightness = _get_random_params(brightness_range)
    return contrast, brightness


def adjust_contrast_brightness(img: np.ndarray, contrast: float, brightness: float):
    '''
    对图像的对比度和亮度进行调整
    :param img: 输入的图像，要求值域为[0, 255]
    :param contrast: 调整对比度，值域为(-1., +inf)
    :param brightness: 调整亮度，值域为(-1., +inf)
    :return:
    '''
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    img = np.asarray(img, np.float32) / 255.

    img = contrast * img + brightness
    img = np.clip(img, 0, 1.) * 255.
    img = img.astype(np.uint8)

    return img


def random_adjust_contrast_brightness(img, contrast_range: Union[float, Iterable]=0., brightness_range: Union[float, Iterable]=0.):
    '''
    在图像的HSL空间进行调整
    :param img: 输入的图像
    :param contrast_range: 调整对比度，值域为(-1., +inf)
    :param brightness_range: 调整亮度，值域为(-1., +inf)
    :return:
    '''
    params = random_params_adjust_contrast_brightness(contrast_range, brightness_range)
    img = adjust_contrast_brightness(img, *params)
    return img


def sp_noise(img, prob):
    '''
    加入固定强度的椒盐噪音
    :param img:
    :param prob:
    :return:
    '''
    assert 0 <= prob <= 1
    im_prob = np.random.uniform(0, 1, size=img.shape).astype(np.float32)
    bad_img = np.random.randint(0, 2, size=img.shape, dtype=img.dtype) * 255
    img = np.where(im_prob < prob, bad_img, img)
    return img


def random_sp_noise(img, prob_range=0.):
    '''
    随机加入不同强度的椒盐噪音
    :param img:
    :param prob_range:  噪音概率
    :return:
    '''
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    prob = _get_random_params(prob_range, not_allow_negative_num=True)
    img = sp_noise(img, prob)
    return img


def gasuss_noise(img, mean, std):
    '''
    加入高斯噪音
    :param img:
    :param mean:
    :param std:
    :return:
    '''
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    noise_im = (np.random.normal(mean, std, size=img.shape) * 255).astype(np.int16)
    img = np.asarray(img, np.int16)
    img += noise_im
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img


def random_gasuss_noise(img, mean_range=0, std_range=0):
    '''
    随机加入不同强度的高斯噪音
    :param img:
    :param mean_range: 均差范围
    :param std_range:  标准差范围
    :return:
    '''
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    mean = _get_random_params(mean_range, not_allow_negative_num=False)
    std = _get_random_params(std_range, not_allow_negative_num=True)
    img = gasuss_noise(img, mean, std)
    return img


if __name__ == '__main__':
    import imageio
    import os
    import im_tool

    impath = os.path.dirname(__file__) + '/laska.png'
    im = imageio.imread(impath)
    im = im[:, :, :3]

    for _ in range(3):
        im1 = random_adjust_HSV(im, 0.3, 0.2, 0.2)
        im_tool.show_image(im1)

    for _ in range(3):
        im1 = random_adjust_contrast_brightness(im, 0.2, 0.5)
        im_tool.show_image(im1)

    for _ in range(3):
        im1 = random_sp_noise(im, 0.3)
        im_tool.show_image(im1)

    for _ in range(3):
        im1 = random_gasuss_noise(im, 0.2, 0.3)
        im_tool.show_image(im1)
