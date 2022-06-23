'''
从病理图中获得主要组织区域的轮廓。
'''

import cv2
import numpy as np
try:
    from .contour_tool import find_contours, draw_contours, resize_contours
except (ModuleNotFoundError, ImportError):
    from contour_tool import find_contours, draw_contours, resize_contours


def get_tissue_contours_default_postprocess(tim: np.ndarray):
    '''
    默认的轮廓布尔图后处理函数
    :param tim:
    :return:
    '''
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cv2.dilate(tim, k, dst=tim, iterations=2)
    cv2.erode(tim, k, dst=tim, iterations=4)
    cv2.dilate(tim, k, dst=tim, iterations=2)
    return tim


def get_tissue_contours(im,
                        gray_thresh=210,
                        channel_diff_thresh=18,
                        area_thresh=0.005,
                        *,
                        postprocess_func=get_tissue_contours_default_postprocess,
                        debug_show=False):
    """
    从一张图像中获得组织轮廓
    :param im:                  opsl缩略图，要求为RGB的nd.ndarray类型
    :param gray_thresh:         灰度差异，像素的任意通道值小于该值，该像素认为是组织
    :param channel_diff_thresh: 通道差异，像素的全部通道的最大值减最小值得到通道差异若大于该值，该像素认为是组织
    :param area_thresh:         面积阈值，组织区域占全图百分比若大于等于该值，则保留该组织区域
    :param postprocess_func:    组织布尔图的后处理函数
    :param debug_show:          是否显示调试数据
    :return: tissue_contours
    """
    assert isinstance(im, np.ndarray)
    assert gray_thresh is not None or channel_diff_thresh is not None, 'Error! Only one can be None between gray_thresh and channel_diff_thresh'
    imhw = im.shape[:2]

    if gray_thresh is None:
        tim1 = np.ones(imhw, dtype=np.uint8)
    else:
        tim1 = np.any(im <= gray_thresh, 2).astype(np.uint8)

    if channel_diff_thresh is None:
        tim2 = np.ones(imhw, dtype=np.uint8)
    else:
        tim2 = (np.max(im, 2) - np.min(im, 2) > channel_diff_thresh).astype(np.uint8)

    tim = tim1 * tim2

    # gim = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    # tim = (gim < gray_thresh).astype(np.uint8)

    if debug_show:
        cv2.imshow('get_tissue_ori', im[..., ::-1])
        # cv2.imshow('get_tissue_gim', gim)
        cv2.imshow('get_tissue_mult', tim*255)
        cv2.imshow('get_tissue_gray', tim1*255)
        cv2.imshow('get_tissue_channel_diff', tim2*255)

    tim = postprocess_func(tim)

    if debug_show:
        cv2.imshow('get_tissue_postprocess', tim*255)

    contours = find_contours(tim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tissue_contours = []

    if area_thresh is None:
        tissue_contours = contours
    else:
        for i, cont in enumerate(contours):
            contour_area = cv2.contourArea(cont)
            factor = float(contour_area) / np.prod(im.shape[:2], dtype=np.float)
            if debug_show:
                print(contour_area, '{:.3f}'.format(factor))
            if factor > area_thresh:
                tissue_contours.append(cont)

    if debug_show:
        mask = np.zeros([im.shape[0], im.shape[1], 1], dtype=np.uint8)
        mask = draw_contours(mask, tissue_contours, [1], thickness=-1)
        cv2.imshow('mask', mask*255)

    return tissue_contours


def get_tissue_contours_with_big_pic(opsl_im,
                                     gray_thresh=210,
                                     channel_diff_thresh=18,
                                     area_thresh=0.005,
                                     *,
                                     postprocess_func=get_tissue_contours_default_postprocess,
                                     thumb_size=768,
                                     debug_show=False):
    '''
    对大图获得组织轮廓，默认参数即可对HE图像工作良好
    灰度差异与通道差异与面积阈值 是与关系
    可以单独设定 gray_thresh 或 channel_diff_thresh 或 area_thresh 为 None，代表关闭对应过滤
    gray_thresh 和 channel_diff_thresh 必须至少有一个不为None
    :param opsl_im:             OpenSlide或兼容图像类型
    :param gray_thresh:         灰度差异，像素的任意通道值小于该值，该像素认为是组织
    :param channel_diff_thresh: 通道差异，像素的全部通道的最大值减最小值得到通道差异若大于该值，该像素认为是组织
    :param area_thresh:         面积阈值，组织区域占全图百分比若大于等于该值，则保留该组织区域
    :param postprocess_func:    组织布尔图的后处理函数
    :param thumb_size:          计算大图轮廓时，取的缩略图的最长边的大小
    :param debug_show:          设定为True可以启动调试功能
    :return:
    '''
    thumb = opsl_im.get_thumbnail([thumb_size, thumb_size])
    thumb = np.array(thumb)
    tissue_contours = get_tissue_contours(thumb, gray_thresh=gray_thresh, channel_diff_thresh=channel_diff_thresh, area_thresh=area_thresh,
                                          postprocess_func=postprocess_func, debug_show=debug_show)
    thumb_hw = thumb.shape[:2]
    factor_hw = np.array(opsl_im.level_dimensions[0][::-1], dtype=np.float32) / thumb_hw
    resized_contours = resize_contours(tissue_contours, factor_hw)
    return resized_contours


def get_custom_contours_with_big_pic(opsl_im, thumb_size, get_contours_func, **get_contours_func_kwargs):
    '''
    对大图获得组织轮廓，使用自定义函数
    :param opsl_im:             OpenSlide或兼容图像类型
    :param get_contours_func:   自定义获取轮廓的函数
    :param get_contours_func_kwargs: 自定义获取轮廓的函数的参数
    :return:
    '''
    thumb = opsl_im.get_thumbnail([thumb_size, thumb_size])
    thumb = np.array(thumb)
    tissue_contours = get_contours_func(thumb, **get_contours_func_kwargs)
    thumb_hw = thumb.shape[:2]
    factor_hw = np.array(opsl_im.level_dimensions[0][::-1], dtype=np.float32) / thumb_hw
    resized_contours = resize_contours(tissue_contours, factor_hw)
    return resized_contours


if __name__ == '__main__':
    import glob
    import os
    if os.name == 'nt':
        openslide_bin = os.path.split(__file__)[0]+'/../bin_openslide_x64_20171122'
        if os.path.isdir(openslide_bin):
            os.add_dll_directory(openslide_bin)
    import openslide as opsl

    for im_path in glob.glob('dataset/ims/*.ndpi', recursive=True):
        im = opsl.OpenSlide(im_path)
        get_tissue_contours_with_big_pic(im, gray_thresh=210, channel_diff_thresh=4, area_thresh=0.001, thumb_size=1024, debug_show=True)
        cv2.waitKey(0)
