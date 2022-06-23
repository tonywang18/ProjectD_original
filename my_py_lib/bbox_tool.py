'''
包围框工具箱，注意，输入和输出的包围框格式均为 y1x1y2x2
如果需要包围框格式转换，请使用 coord_tool 转换坐标格式
'''


import numpy as np
from typing import Sized, Union
try:
    from .numpy_tool import universal_get_shortest_link_pair
except ImportError:
    from numpy_tool import universal_get_shortest_link_pair


def check_bbox_or_bboxes(bbox_or_bboxes):
    '''
    检查输入是否为单个bbox或一组bbox
    检查输入是否符合要求
    :param bbox_or_bboxes:
    :return:
    '''
    assert isinstance(bbox_or_bboxes, np.ndarray)
    assert bbox_or_bboxes.ndim in [1, 2] and bbox_or_bboxes.shape[-1] == 4


def check_bbox(bbox):
    '''
    检查输入是否为单个bbox
    检查输入是否符合要求
    :param bbox_or_bboxes:
    :return:
    '''
    assert isinstance(bbox, np.ndarray)
    assert bbox.ndim in [1,] and bbox.shape[-1] == 4


def check_bboxes(bboxes):
    '''
    检查输入是否为单个bbox
    检查输入是否符合要求
    :param bbox_or_bboxes:
    :return:
    '''
    assert isinstance(bboxes, np.ndarray)
    assert bboxes.ndim in [2,] and bboxes.shape[-1] == 4


def calc_bboxes_area(bbox_or_bboxes: np.ndarray):
    '''
    求轮廓的面积
    :param bbox_or_bboxes:      要求 bboxes 为 np.ndarray 和 格式为 y1x1y2x2
    :return:
    '''
    check_bbox_or_bboxes(bbox_or_bboxes)
    hw = bbox_or_bboxes[..., 2:] - bbox_or_bboxes[..., :2]
    area = hw[..., 0] * hw[..., 1]
    return area


def offset_bboxes(bbox_or_bboxes, ori_yx=(0, 0), new_ori_yx=(0, 0)):
    '''
    重定位包围框位置
    :param bbox_or_bboxes:      要求 bboxes 为 np.ndarray 和 格式为 y1x1y2x2
    :param ori_yx:              旧起点
    :param new_ori_yx:          新起点
    :return:
    '''
    bboxes = bbox_or_bboxes

    check_bbox_or_bboxes(bboxes)
    assert len(ori_yx) == 2
    assert len(new_ori_yx) == 2

    bboxes = np.copy(bboxes)

    bboxes[..., :2] = bboxes[..., :2] - ori_yx
    bboxes[..., 2:] = bboxes[..., 2:] - ori_yx

    bboxes[..., :2] = bboxes[..., :2] + new_ori_yx
    bboxes[..., 2:] = bboxes[..., 2:] + new_ori_yx

    return bboxes


def resize_bboxes(bbox_or_bboxes: np.ndarray, factor_hw=(1, 1), center_yx=(0, 0)):
    '''
    基于指定位置对包围框进行缩放
    :param bboxes:      要求 bboxes 为 np.ndarray 和 格式为 y1x1y2x2
    :param factor_hw:   缩放倍率，可以单个数字或元组
    :param center_yx:   默认以(0, 0)为原点进行缩放
    :return:
    '''
    bboxes = bbox_or_bboxes
    check_bbox_or_bboxes(bboxes)

    if not isinstance(factor_hw, Sized):
        factor_hw = [float(factor_hw), float(factor_hw)]

    assert len(center_yx) == 2
    assert len(factor_hw) == 2

    center_yxyx = np.array([*center_yx] * 2)
    factor_hwhw = np.array([*factor_hw] * 2)

    ori_dtype = bboxes.dtype

    bboxes = np.asarray(bboxes, np.float32)
    bboxes -= center_yxyx
    bboxes *= factor_hwhw
    bboxes += center_yxyx
    bboxes = np.asarray(bboxes, ori_dtype)

    return bboxes


def calc_bboxes_center(bbox_or_bboxes: np.ndarray):
    '''
    求单个或多个包围框的中心坐标
    :param bbox_or_bboxes: 要求 bbox 为 np.ndarray 和 格式为 y1x1y2x2
    :return:
    '''
    bboxes = bbox_or_bboxes
    check_bbox_or_bboxes(bboxes)
    centers = np.asarray((bboxes[..., :2] + bboxes[..., 2:]) / 2, dtype=bboxes.dtype)
    return centers


calc_bbox_center = calc_bboxes_center


def inter_bbox_1to1(bbox1, bbox2):
    '''
    计算一个包围框与包围框组的相交包围框。当相交区域不存在时，返回None
    :param bbox1:
    :param bbox2:
    :return:
    '''
    check_bbox(bbox1)
    check_bbox(bbox2)

    inter_y1 = np.maximum(bbox1[0], bbox2[0])
    inter_x1 = np.maximum(bbox1[1], bbox2[1])
    inter_y2 = np.minimum(bbox1[2], bbox2[2])
    inter_x2 = np.minimum(bbox1[3], bbox2[3])

    if inter_y2 > inter_y1 and inter_x2 > inter_x1:
        inter_bbox = np.asarray([inter_y1, inter_x1, inter_y2, inter_x2], dtype=bbox1.dtype)
    else:
        inter_bbox = None

    return inter_bbox


def calc_bbox_occupancy_ratio_1toN_or_Nto1(bboxes1: np.ndarray, bboxes2: np.ndarray):
    '''
    计算bboxes1与bboxes2相交区域与bboxes2的区域的占比，下面使用了...技巧，使其同时支持 1对1，1对N，N对1 的IOU计算
    注意不支持 N对M 的计算
    :param bboxes1:
    :param bboxes2:
    :return:
    '''
    # bboxes1_area = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    bboxes2_area = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    inter_y1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    inter_x1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    inter_y2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    inter_x2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    # 计算相交区域长宽
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_area = inter_h * inter_w

    # uniou_area = bboxes1_area + bboxes2_area - inter_area
    # 如果分母为0时，使其等于1e-8，可以同时避免eps对iou干扰和除零问题
    bboxes2_area = np.where(bboxes2_area > 0, bboxes2_area, 1e-8)

    ious = inter_area / bboxes2_area
    return ious


def calc_bbox_occupancy_ratio_1toN(bbox1, bboxes):
    '''
    计算包围框1和包围框组的相交区域与包围框组的面积的占比
    :param bbox1:
    :param bboxes:
    :return:
    '''
    check_bbox(bbox1)
    check_bboxes(bboxes)

    ratio = calc_bbox_occupancy_ratio_1toN_or_Nto1(bbox1, bboxes)
    return ratio


def calc_bbox_occupancy_ratio_Nto1(bboxes, bbox2):
    '''
    计算包围框组和包围框2的相交区域与包围框2的面积的占比
    :param bboxes:
    :param bbox2:
    :return:
    '''
    check_bboxes(bboxes)
    check_bbox(bbox2)

    ratio = calc_bbox_occupancy_ratio_1toN_or_Nto1(bboxes, bbox2)
    return ratio


def calc_bbox_occupancy_ratio_1to1(bbox1, bbox2):
    '''
    计算包围框1和包围框2的相交区域与包围框2的面积的占比
    :param bbox1:
    :param bbox2:
    :return:
    '''
    check_bbox(bbox1)
    check_bbox(bbox2)

    ratio = calc_bbox_occupancy_ratio_1toN_or_Nto1(bbox1, bbox2)
    return ratio


def _calc_bbox_iou_1to1_or_1toN_or_Nto1(bboxes1: np.ndarray, bboxes2: np.ndarray):
    '''
    计算N对个包围框的IOU，注意这里一一对应的，同时下面使用了...技巧，使其同时支持 1对1，1对N，N对1 的IOU计算
    注意不支持 N对M 的计算
    :param bboxes1:
    :param bboxes2:
    :return:
    '''
    bboxes1_area = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    bboxes2_area = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    inter_y1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    inter_x1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    inter_y2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    inter_x2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    # 计算相交区域长宽
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_area = inter_h * inter_w

    uniou_area = bboxes1_area + bboxes2_area - inter_area
    # 如果分母为0时，使其等于1e-8，可以同时避免eps对iou干扰和除零问题
    uniou_area = np.where(uniou_area > 0, uniou_area, 1e-8)

    ious = inter_area / uniou_area
    return ious


def calc_bbox_iou_1toN(bbox1: np.ndarray, bboxes: np.ndarray):
    '''
    计算 1对N 的 IOU
    :param bbox1:
    :param bboxes:
    :return:
    '''
    check_bbox(bbox1)
    check_bboxes(bboxes)
    return _calc_bbox_iou_1to1_or_1toN_or_Nto1(bbox1, bboxes)


def calc_bbox_iou_NtoM(bboxes1: np.ndarray, bboxes2: np.ndarray):
    '''
    计算 包围框组N与包围框组M两两之间 的 IOU
    :param bboxes1:
    :param bboxes2:
    :return:
    '''
    check_bboxes(bboxes1)
    check_bboxes(bboxes2)
    pair_ious = np.zeros([len(bboxes1), len(bboxes2)], np.float32)
    for i in range(len(bboxes1)):
        pair_ious[i] = _calc_bbox_iou_1to1_or_1toN_or_Nto1(bboxes1[i], bboxes2)
    return pair_ious


def pad_bbox_to_square(bbox: np.ndarray):
    '''
    将输入的包围框填充为正方形，要求包围框坐标格式为y1x1y2x2或x1y1x2y2
    :param bbox:
    :return:
    '''
    assert bbox.ndim == 1 and len(bbox) == 4
    dtype = bbox.dtype
    bbox = np.asarray(bbox, np.float32)
    t1 = bbox[2:] - bbox[:2]
    tmax = max(t1)
    t2 = tmax / t1
    c = (bbox[:2] + bbox[2:]) / 2
    bbox = resize_bboxes(bbox, t2, c)
    bbox = np.asarray(bbox, dtype)
    return bbox


def nms_process(confs: np.ndarray, bboxes: np.ndarray, iou_thresh: float=0.7):
    '''
    NMS 过滤，只需要置信度和坐标
    :param confs:       置信度列表，形状为 [-1] 或 [-1, 1]
    :param bboxes:      包围框列表，形状为 [-1, 4]，坐标格式为 y1x1y2x2 或 x1y1x2y2
    :param iou_thresh:  IOU阈值
    :return:
    '''
    assert len(confs) == len(bboxes)
    confs = np.asarray(confs, np.float32).reshape([-1])
    bboxes = np.asarray(bboxes, np.float32).reshape([-1, 4])
    assert len(confs) == len(bboxes)
    ids = np.argsort(confs)[::-1]
    ids = ids.reshape([-1])
    keep_boxes = []
    keep_ids = []
    for i in ids:
        if len(keep_boxes) == 0:
            keep_boxes.append(bboxes[i])
            keep_ids.append(i)
            continue
        cur_box = bboxes[i]
        ious = calc_bbox_iou_1toN(cur_box, np.asarray(keep_boxes, np.float32))
        max_iou = np.max(ious)
        if max_iou > iou_thresh:
            continue
        keep_boxes.append(bboxes[i])
        keep_ids.append(i)
    return keep_ids


def make_bbox_by_center_point(center_yx, bbox_hw: Union[int, float, Sized], dtype=np.float32):
    '''
    基于中心点和宽高生成y1x1y2x2包围框
    :param center_yx: 中心点
    :param bbox_hw: 包围框大小
    :return:
    '''
    if not isinstance(bbox_hw, Sized):
        bbox_hw = [bbox_hw, bbox_hw]

    assert len(center_yx) == 2
    assert len(bbox_hw) == 2

    center_yx = np.asarray(center_yx, np.float32)
    bbox_hw = np.asarray(bbox_hw, np.float32) / 2

    return np.asarray([center_yx[0] - bbox_hw[0], center_yx[1] - bbox_hw[1], center_yx[0] + bbox_hw[0], center_yx[1] + bbox_hw[1]], dtype=dtype)


def _is_points_in_bboxes_1to1_or_1toN_or_Nto1(points, bboxes):
    '''
    点是否在包围盒内部
    :param points:
    :param bboxes:
    :return:
    '''
    b = np.logical_and(np.all(points > bboxes[..., :2], 1), np.all(points < bboxes[..., 2:], 1))
    return b


is_points_in_bbox = _is_points_in_bboxes_1to1_or_1toN_or_Nto1
is_point_in_bboxes = _is_points_in_bboxes_1to1_or_1toN_or_Nto1
is_point_in_bbox = _is_points_in_bboxes_1to1_or_1toN_or_Nto1


def get_bboxes_shortest_link_pair(bboxes1, bboxes2, iou_th: float, *, pair_ious=None):
    '''
    求两组包围框之间最大IOU配对
    :param bboxes1:     包围框组1
    :param bboxes2:     包围框组2
    :param iou_th:      iou阈值
    :param pair_ious:   如果已有现成的配对IOU表，则使用现成的，可以省去生成IOU表的时间。IOU表 可用 calc_iou_with_contours_NtoM 函数生成
    :return: 返回轮廓的编号和他们的IOU
    '''

    if pair_ious is None:
        pair_ious = calc_bbox_iou_NtoM(bboxes1, bboxes2)
    else:
        assert pair_ious.shape == (len(bboxes1), len(bboxes2)), 'Error! Bad param pair_ious.'

    out_contact_ids1, out_contact_ids2, out_contact_iou = universal_get_shortest_link_pair(pair_ious, iou_th, np.greater)
    return out_contact_ids1, out_contact_ids2, out_contact_iou
