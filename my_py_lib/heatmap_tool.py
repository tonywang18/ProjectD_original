'''
热图工具
'''

import numpy as np
import cv2
from skimage.draw import disk as sk_disk
from . import bbox_tool
from . import contour_tool
from typing import Iterable


def get_cls_with_det_pts_from_cls_hm(cls_hm: np.ndarray, det_pts: np.ndarray, radius: int=3, mode: str='mean'):
    '''
    根据检测点和分类热图，从中获得获得每个检测点的类别
    :param cls_hm:  分类热图
    :param det_pts: 检测点坐标，要求形状为 [-1, 2]，坐标格式为 yx
    :param radius:  检索半径
    :param mode:    计算模式，转换检索半径内类别像素的到分类概率的方式
    :return:
    '''
    assert cls_hm.ndim == 3
    assert cls_hm.shape[2] >= 1
    assert mode in ('sum', 'mean', 'max', 'min')
    det_pts = np.asarray(det_pts, np.int32)
    if len(det_pts) == 0:
        det_pts = det_pts.reshape([-1, 2])
    assert det_pts.ndim == 2 and det_pts.shape[1] == 2
    radius = int(radius)
    # 每个点对应各个类别的概率
    cls_probs = np.zeros([len(det_pts), cls_hm.shape[2]], np.float32)
    hw = cls_hm.shape[:2]
    for i, pt in enumerate(det_pts):
        rr, cc = sk_disk(pt, radius=radius, shape=hw)
        for c in range(cls_hm.shape[2]):
            if mode == 'sum':
                cls_probs[i, c] = cls_hm[rr, cc, c].sum()
            elif mode == 'mean':
                cls_probs[i, c] = cls_hm[rr, cc, c].mean()
            elif mode == 'max':
                cls_probs[i, c] = cls_hm[rr, cc, c].max()
            elif mode == 'min':
                cls_probs[i, c] = cls_hm[rr, cc, c].min()
            else:
                raise AssertionError('Error! Invalid mode param.')
    return cls_probs


def tr_bboxes_to_tlbr_heatmap(im_hw, hm_hw, bboxes, n_class=0, classes=None):
    '''
    转换包围框到tlbr热图（FCOS热图方式）
    额外修改，仅在置信度大于0.5时，才会设定tlbr值，否则坐标值为0
    :param im_hw:   原始图像大小
    :param hm_hw:   热图大小
    :param bboxes:  包围框，形状为[-1, 4]，坐标格式为[y1x1y2x2, y1x1y2x2]
    :return:
    '''
    if n_class > 0:
        assert classes is not None, 'Error! The classes is not allow None when n_class > 0.'
        assert len(bboxes) == len(classes), 'Error! The len(bboxes) must be equal len(classes).'
        if len(classes) > 0:
            assert max(classes) < n_class and min(classes) >= 0, 'Error! All class must be in range [0, n_class).'

    ohm = np.zeros([*hm_hw, 1 + 4 + n_class], dtype=np.float32)
    im_hw = np.asarray(im_hw, np.int32)
    hm_hw = np.asarray(hm_hw, np.int32)
    for box_id, box in enumerate(bboxes):
        block_float_box = np.asarray(box / [*im_hw, *im_hw] * [*hm_hw, *hm_hw], np.float32)
        block_box = np.asarray(block_float_box, np.int32)
        center_hw = bbox_tool.calc_bbox_center(block_float_box).astype(np.int32)
        for ih in range(max(block_box[0], 0), min(block_box[2] + 1, hm_hw[0])):
            for iw in range(max(block_box[1], 0), min(block_box[3] + 1, hm_hw[1])):
                # 只有正中间的IOU值，才设定为1，周围都设定为0
                # if ih == block_int_center[0] and iw == block_int_center[1]:
                #     ohm[ih, iw, :1] = 1.
                cur_pos = np.asarray([ih + 0.5, iw + 0.5], np.float32)
                t = cur_pos[0] - block_float_box[0]
                l = cur_pos[1] - block_float_box[1]
                b = block_float_box[2] - cur_pos[0]
                r = block_float_box[3] - cur_pos[1]
                if np.min([t, l, b, r]) <= 0:
                    continue

                # 计算中心度
                if ih == center_hw[0] and iw == center_hw[1]:
                    c = 1.
                else:
                    c = np.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))

                # 设定置信度
                if c == 1.:
                    ohm[ih, iw, 0:1] = c
                elif ohm[ih, iw, 0:1] > 0:
                    ohm[ih, iw, 0:1] = max(ohm[ih, iw, 0:1], c) - min(ohm[ih, iw, 0:1], c)
                else:
                    ohm[ih, iw, 0:1] = c

                # 设定分类
                if n_class > 0:
                    ohm[ih, iw, 5+classes[box_id]] = c

                # yx位置，同样设定
                if c > 0.5:
                    ohm[ih, iw, 1:5] = [t, l, b, r]
    return ohm


def tr_tlbr_heatmap_to_bboxes(ohm, im_hw, thresh=0.5):
    '''
    转换tlbr热图（FCOS热图方式）到包围框
    :param ohm:     tlbr热图块输入，要求格式为 [H,W,1+4]
    :param im_hw:   原图像大小
    :param thresh:  阈值
    :return:
    '''
    assert isinstance(ohm, np.ndarray) and ohm.ndim == 3 and ohm.shape[2] >= 5
    assert len(im_hw) == 2

    has_class = ohm.shape[2] > 5
    n_class = ohm.shape[2] - 5

    hm_hw = ohm.shape[:2]
    hm_hwhw = np.asarray([*hm_hw, *hm_hw], np.float32)
    im_hwhw = np.asarray([*im_hw, *im_hw], np.float32)
    confs, tlbr, classes = np.split(ohm, [1, 5], 2)
    grid = np.transpose(np.mgrid[:hm_hw[0], :hm_hw[1]], [1, 2, 0])
    bs = confs > thresh

    s_confs = confs[bs]
    if has_class:
        s_classes = classes[bs[..., 0]]

    s_tlbr = tlbr[bs[..., 0]]
    s_grid = grid[bs[..., 0]] + 0.5
    s_tlbr[:, :2] = s_grid - s_tlbr[:, :2]
    s_tlbr[:, 2:] = s_grid + s_tlbr[:, 2:]
    bboxes = s_tlbr / hm_hwhw * im_hwhw
    s_confs = np.asarray(s_confs, np.float32).reshape([-1, 1])
    bboxes = np.asarray(bboxes, np.float32).reshape([-1, 4])

    if has_class:
        s_classes = np.asarray(s_classes, np.float32).reshape([-1, n_class])
        return s_confs, bboxes, s_classes
    else:
        return s_confs, bboxes


def get_cls_contours_from_cls_hm(cls_hm, probs=0.5):
    '''
    从类别热图中获得多个类别轮廓
    :param cls_hm:
    :param probs:
    :return:
    '''
    assert cls_hm.ndim == 3
    assert cls_hm.shape[2] >= 1

    C = cls_hm.shape[-1]

    if not isinstance(probs, Iterable):
        probs = [probs] * C

    cls = []
    cs = []

    for c in range(C):
        bin = np.uint8(cls_hm[..., c] > probs[c])
        conts = contour_tool.find_contours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, keep_invalid_contour=False)
        n = len(conts)
        cls.extend([c]*n)
        cs.extend(conts)

    return cls, cs
