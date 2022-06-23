'''
opencv 的单个轮廓格式为 [N, 1, xy]
我的单个轮廓格式为 [N, yx]

下面所有函数，除了轮廓格式转换函数和部分函数，都是用我的轮廓格式输入和输出
函数名以shapely开头的，需要输入polygon格式

注意，下面函数需要的坐标要求为整数，不支持浮点数运算
因为下面一些计算需要画图，浮点数无法画图。

现在，可以支持浮点数了，现在使用shapely库用作轮廓运算

优先使用opencv的函数处理，opencv没有的才使用shapely，因为opencv函数的速度比shapely快

'''

import cv2
assert cv2.__version__ >= '4.0'
import numpy as np
from typing import Iterable, List
from shapely.geometry import Polygon, MultiPolygon, MultiPoint, Point, LineString, LinearRing
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import unary_union
import warnings
try:
    from im_tool import ensure_image_has_same_ndim
    from list_tool import list_multi_get_with_ids, list_multi_get_with_bool
    import point_tool
    from numpy_tool import universal_get_shortest_link_pair
except ModuleNotFoundError:
    from .im_tool import ensure_image_has_same_ndim
    from .list_tool import list_multi_get_with_ids, list_multi_get_with_bool
    from . import point_tool
    from .numpy_tool import universal_get_shortest_link_pair


def check_and_tr_umat(mat):
    '''
    如果输入了cv.UMAT，将会自动转换为 np.ndarray
    :param mat:
    :return:
    '''
    if isinstance(mat, cv2.UMat):
        mat = mat.get()
    return mat


def tr_cv_to_my_contours(cv_contours):
    '''
    轮廓格式转换，转换opencv格式到我的格式
    :param cv_contours:
    :return:
    '''
    out_contours = [c[:, 0, ::-1] for c in cv_contours]
    return out_contours


def tr_my_to_cv_contours(my_contours):
    '''
    轮廓格式转换，转换我的格式到opencv的格式
    :param my_contours:
    :return:
    '''
    out_contours = [c[:, None, ::-1] for c in my_contours]
    return out_contours


def tr_my_to_polygon(my_contours):
    '''
    轮廓格式转换，转换我的格式到polygon
    :param my_contours:
    :return:
    '''
    polygons = []
    for c in my_contours:
        # 如果c是float64位的，要转化为float32位
        if c.dtype == np.float64:
            c = c.astype(np.float32)
        # 如果点数少于3个，就先转化为多个点，然后buffer(1)转化为轮廓，可能得到MultiPolygon，使用convex_hull得到凸壳
        if len(c) < 3 or calc_contour_area(c) == 0:
            p = MultiPoint(c[:, ::-1]).buffer(1).convex_hull
        else:
            p = Polygon(c[:, ::-1])
        if not p.is_valid:
            # 如果轮廓在buffer(0)后变成了MultiPolygon，则尝试buffer(1)，如果仍然不能转换为Polygon，则将轮廓转换为凸壳，强制转换为Polygon
            p1 = p.buffer(0)
            if not isinstance(p1, Polygon):
                p1 = p.buffer(1)
            if not isinstance(p1, Polygon):
                warnings.warn('Warning! Found an abnormal contour that cannot be converted directly to Polygon, currently will be forced to convex hull to allow it to be converted to Polygon')
                p1 = p.convex_hull
            p = p1
        polygons.append(p)
    return polygons


def tr_polygons_to_my(polygons: List[Polygon], dtype: np.dtype=np.float32):
    '''
    转换shapely的多边形到我的格式
    :param polygons:
    :param dtype: 输出数据类型
    :return:
    '''
    my_contours = []
    tmp = shapely_ensure_polygon_list(polygons)
    assert len(tmp) == len(polygons), 'Error! The input polygons has some not Polygon item.'
    polygons = tmp
    for poly in polygons:
        x, y = poly.exterior.xy
        c = np.array(list(zip(y, x)), dtype)
        my_contours.append(c)
    return my_contours


def tr_polygons_to_cv(polygons: List[Polygon], dtype=np.float32):
    '''
    转换shapely的多边形到opencv格式
    :param polygons:
    :param dtype:   输出数据类型
    :return:
    '''
    cv_contours = []
    for poly in polygons:
        x, y = poly.exterior.xy
        c = np.array(list(zip(x, y)), dtype)[:, None]
        cv_contours.append(c)
    return cv_contours


def tr_my_to_linering(my_contours):
    '''
    轮廓格式转换，转换我的格式到linering
    :param my_contours:
    :return:
    '''
    ps = tr_my_to_polygon(my_contours)
    linerings = [p.exterior for p in ps]
    return linerings


def tr_linering_to_my(linerings: List[LineString], dtype=np.float32):
    '''
    轮廓格式转换，转换linering到我的格式
    :param my_contours:
    :return:
    '''
    cs = []
    for lr in linerings:
        xs = np.asarray(lr.xy[0].tolist(), dtype=dtype)
        ys = np.asarray(lr.xy[1].tolist(), dtype=dtype)
        c = np.stack([ys, xs], 1)
        cs.append(c)
    return cs


def shapely_ensure_polygon_list(ps):
    '''
    确保返回值是 List[Polygon]
    本函数目的是减少shapely返回值可能是任何类的问题
    :param ps:
    :return:
    '''
    if isinstance(ps, Polygon):
        ps = [ps]
    elif isinstance(ps, MultiPolygon):
        ps = list(ps.geoms)
    elif isinstance(ps, BaseMultipartGeometry):
        ps = [p for p in ps.geoms if isinstance(p, Polygon)]
    elif isinstance(ps, Iterable):
        ps = [p for p in ps if isinstance(p, Polygon)]
    elif isinstance(ps, (Point, LineString, LinearRing)):
        ps = []
    else:
        raise RuntimeError('Error! Bad input in shapely_ensure_polygon_list.', str(type(ps)), ps)
    return ps


def find_contours(im, mode, method, keep_invalid_contour=False):
    '''
    cv2.findContours 的包装，区别是会自动转换cv2的轮廓格式到我的格式，和会自动删除轮廓点少于3的无效轮廓
    :param im:
    :param mode: 轮廓查找模式，例如 cv2.RETR_EXTERNAL cv2.RETR_TREE, cv2.RETR_LIST
    :param method: 轮廓优化方法，例如 cv2.CHAIN_APPROX_SIMPLE cv2.CHAIN_APPROX_NONE
    :param keep_invalid_contour: 是否保留轮廓点少于3的无效轮廓。
    :return:
    '''
    # 简化轮廓，目前用于缩放后去除重叠点，并不进一步简化
    # epsilon=0 代表只去除重叠点
    contours, _ = cv2.findContours(im, mode=mode, method=method)
    # 删除轮廓点少于3的无效轮廓
    valid_contours = []
    if not keep_invalid_contour:
        for c in contours:
            if len(c) >= 3:
                valid_contours.append(c)
    else:
        valid_contours.extend(contours)
    valid_contours = tr_cv_to_my_contours(valid_contours)
    return valid_contours


def offset_contours(contours, ori_yx=(0, 0), new_ori_yx=(0, 0)):
    '''
    重定位轮廓位置
    :param contours:    多个轮廓，要求为我的格式
    :param ori_yx:      旧轮廓的起点
    :param new_ori_yx:  新起点
    :return:
    '''
    if len(contours) == 0:
        return contours
    new_contours = []
    ori_yx = np.array(ori_yx, dtype=contours[0].dtype).reshape(1, 2)
    new_ori_yx = np.array(new_ori_yx, dtype=contours[0].dtype).reshape(1, 2)
    for c in contours:
        c = c - ori_yx + new_ori_yx
        new_contours.append(c)
    return new_contours


def make_contour_from_bbox(bbox):
    '''
    将包围框转换为轮廓
    :param bbox: np.ndarray [y1, x1, y2, x2]
    :return:
    '''
    p1 = [bbox[0], bbox[1]]
    p2 = [bbox[0], bbox[3]]
    p3 = [bbox[2], bbox[3]]
    p4 = [bbox[2], bbox[1]]
    contours = [p1, p2, p3, p4]
    contours = np.array(contours)
    return contours


def make_bbox_from_contour(contour):
    '''
    求轮廓的外接包围框
    :param contour:
    :return:
    '''
    min_y = np.min(contour[:, 0])
    max_y = np.max(contour[:, 0])
    min_x = np.min(contour[:, 1])
    max_x = np.max(contour[:, 1])
    bbox = np.array([min_y, min_x, max_y, max_x])
    return bbox


def simple_contours(contours, epsilon=0):
    '''
    简化轮廓，当俩个点的距离小于等于 epsilon 时，会融合这俩个点。
    epsilon=0 代表只消除重叠点
    :param contours:
    :param epsilon:
    :return:
    '''
    # 简化轮廓，目前用于缩放后去除重叠点，并不进一步简化
    # epsilon=0 代表只去除重叠点
    contours = tr_my_to_cv_contours(contours)
    out = []
    for c in contours:
        out.append(cv2.approxPolyDP(c, epsilon, True))
    out = tr_cv_to_my_contours(out)
    return out


def _resize_contour(contour, scale_factor_hw, offset_yx, dtype):
    return ((contour - offset_yx) * scale_factor_hw + offset_yx).astype(dtype)


def resize_contour(contour, scale_factor_hw=1.0, offset_yx=(0, 0)):
    '''
    缩放轮廓
    :param contour: 输入一个轮廓
    :param scale_factor_hw: 缩放倍数
    :param offset_yx: 偏移位置，默认为左上角(0, 0)
    :return:
    '''
    assert isinstance(contour, np.ndarray) and contour.ndim == 2 and contour.shape[-1] == 2
    scale_factor_hw = np.asarray(scale_factor_hw).reshape([1, -1])
    offset_yx = np.asarray(offset_yx, np.float32).reshape([1, 2])
    out = _resize_contour(contour, scale_factor_hw, offset_yx, contour.dtype)
    return out


def resize_contours(contours, scale_factor_hw=1.0, offset_yx=(0, 0)):
    '''
    缩放轮廓
    :param contours: 输入一组轮廓
    :param scale_factor_hw: 缩放倍数
    :param offset_yx: 偏移位置，默认为左上角(0, 0)
    :return:
    '''
    scale_factor_hw = np.asarray(scale_factor_hw).reshape([1, -1])
    offset_yx = np.asarray(offset_yx, np.float32).reshape([1, 2])
    out = [_resize_contour(c, scale_factor_hw, offset_yx, contours[0].dtype) for c in contours]
    return out


def calc_contour_area(contour: np.ndarray):
    '''
    求轮廓的面积
    :param contour: 输入一个轮廓
    :return:
    '''
    area = cv2.contourArea(tr_my_to_cv_contours([contour])[0])
    return area


def calc_contours_area(contours: np.ndarray):
    '''
    求一组轮廓的面积
    :param contours: 输入一组轮廓
    :return:
    '''
    cs = tr_my_to_cv_contours(contours)
    areas = [cv2.contourArea(c) for c in cs]
    return areas


def calc_convex_contours(coutours):
    '''
    计算一组轮廓的凸壳轮廓，很多时候可以加速。
    :param coutours: 一组轮廓
    :return: 返回一组自身的凸壳轮廓
    '''
    coutours = tr_my_to_cv_contours(coutours)
    new_coutours = [cv2.convexHull(con) for con in coutours]
    new_coutours = tr_cv_to_my_contours(new_coutours)
    return new_coutours


def shapely_calc_distance_polygon(polygon1: Polygon, polygon2: Polygon):
    '''
    求俩个轮廓的最小距离，原型
    如果一个轮廓与另一个轮廓相交或在其内部，距离将为0
    :param polygon1: 多边形1
    :param polygon2: 多边形2
    :return: 轮廓距离
    '''
    assert isinstance(polygon1, Polygon)
    assert isinstance(polygon2, Polygon)
    l = polygon1.distance(polygon2)
    return l


def shapely_calc_distance_linering(linering1: LineString, linering2: LineString):
    '''
    求俩个线环的最小距离，原型
    如果线环在另外一个线环的内部，但不相交，此时返回它们的最接近点的距离，而不是返回0
    :param polygon1: 多边形1
    :param polygon2: 多边形2
    :return: 轮廓面积
    '''
    assert isinstance(linering1, LineString)
    assert isinstance(linering2, LineString)
    l = linering1.distance(linering2)
    return l


def calc_distance_polygon_1toN(cont1: np.ndarray, conts: list):
    '''
    求俩个轮廓的最小距离
    如果一个轮廓与另一个轮廓相交或在其内部，距离将为0
    :param cont1: 轮廓1，我的格式
    :param conts: 轮廓组
    :return: 轮廓距离
    '''
    p1 = tr_my_to_polygon([cont1])[0]
    ps = tr_my_to_polygon(conts)
    ds = np.zeros([len(conts)], np.float32)
    for i, p in enumerate(ps):
        ds[i] = shapely_calc_distance_polygon(p1, p)
    return ds


def calc_distance_linering_1toN(cont1: np.ndarray, conts: list):
    '''
    求俩个线环的最小距离
    如果线环在另外一个线环的内部，但不相交，此时返回它们的最接近点的距离，而不是返回0
    :param cont1: 轮廓1，我的格式
    :param conts: 轮廓组
    :return: 轮廓面积
    '''
    r1 = tr_my_to_linering([cont1])[0]
    rs = tr_my_to_linering(conts)
    ds = np.zeros([len(conts)], np.float32)
    for i, r in enumerate(rs):
        ds[i] = shapely_calc_distance_linering(r1, r)
    return ds


# def calc_iou_with_two_contours(contour1, contour2, max_test_area_hw=(512, 512)):
#     '''
#     求两个轮廓的IOU分数，使用绘图法，速度相当慢。
#     :param contour1: 轮廓1
#     :param contour2: 轮廓2
#     :param max_test_area_hw: 最大缓存区大小，若计算的轮廓大小大于限制，则会先缩放轮廓
#     :return:
#     '''
#     # 计算俩个轮廓的iou
#     bbox1 = calc_bbox_with_contour(contour1)
#     bbox2 = calc_bbox_with_contour(contour2)
#     merge_bbox_y1x1 = np.where(bbox1 < bbox2, bbox1, bbox2)[:2]
#     merge_bbox_y2x2 = np.where(bbox1 > bbox2, bbox1, bbox2)[2:]
#     contour1 = contour1 - merge_bbox_y1x1
#     contour2 = contour2 - merge_bbox_y1x1
#     merge_bbox_y2x2 -= merge_bbox_y1x1
#     merge_bbox_y1x1.fill(0)
#     merge_bbox_hw = merge_bbox_y2x2
#     if max_test_area_hw is not None:
#         hw_factor = merge_bbox_hw / max_test_area_hw
#         max_factor = np.max(hw_factor)
#         if max_factor > 1:
#             contour1, contour2 = resize_contours([contour1, contour2], 1 / max_factor)
#             merge_bbox_hw = np.ceil(merge_bbox_hw / max_factor).astype(np.int32)
#     reg = np.zeros(merge_bbox_hw.astype(np.int), np.uint8)
#     reg1 = draw_contours(reg.copy(), [contour1], 1, -1)
#     reg2 = draw_contours(reg.copy(), [contour2], 1, -1)
#     cross_reg = np.all([reg1, reg2], 0).astype(np.uint8)
#     cross_contours, _ = cv2.findContours(cross_reg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     cross_area = 0
#     for c in cross_contours:
#         cross_area += cv2.contourArea(c)
#     contour1_area = calc_contour_area(contour1)
#     contour2_area = calc_contour_area(contour2)
#     iou = cross_area / (contour1_area + contour2_area - cross_area + 1e-8)
#     return iou


def shapely_calc_iou_with_two_contours(contour1, contour2):
    '''
    计算俩个轮廓的IOU，原型
    :param contour1: polygon多边形1
    :param contour2: polygon多边形1
    :return:    IOU分数
    '''
    c1 = contour1
    c2 = contour2
    if not c1.intersects(c2):
        return 0.
    area1 = c1.area
    area2 = c2.area
    inter_area = c1.intersection(c2).area
    iou = inter_area / (area1 + area2 - inter_area + 1e-8)
    return iou


def calc_iou_with_two_contours(contour1, contour2):
    '''
    计算两个轮廓的IOU
    :param contour1: 轮廓1
    :param contour2: 轮廓2
    :return:
    '''
    c1, c2 = tr_my_to_polygon([contour1, contour2])
    iou = shapely_calc_iou_with_two_contours(c1, c2)
    return iou


def draw_contours(im, contours, color, thickness=2, copy=False):
    '''
    绘制轮廓
    :param im:  图像
    :param contours: 多个轮廓的列表，格式为 [(n,pt_yx), (n,pt_yx)]
    :param color: 绘制的颜色
    :param thickness: 绘制轮廓边界大小
    :param copy: 默认在原图上绘制
    :return:
    '''
    ori_im = im
    if isinstance(color, Iterable):
        color = tuple(color)
    else:
        color = (color,)
        if im.ndim == 3:
            color = color * im.shape[-1]
    contours = tr_my_to_cv_contours(contours)
    # 确保为整数
    contours = [np.int32(c) for c in contours]
    if copy:
        im = im.copy()
    else:
        # 必须是连续可写数组才能被opencv原地写入
        assert im.flags['C_CONTIGUOUS'] and im.flags['WRITEABLE'], 'Error! Draw im is not C_CONTIGUOUS or not WRITEABLE.'
    im = cv2.drawContours(im, contours, -1, color, thickness)
    im = check_and_tr_umat(im)
    im = ensure_image_has_same_ndim(im, ori_im)
    return im


def shapely_calc_distance_with_contours_1toN(c1: Polygon, batch_c: Iterable[Polygon]):
    '''
    求一个轮廓和一组轮廓的IOU分数，原型
    :param c1:
    :param batch_c:
    :return:
    '''
    ious = np.zeros([len(batch_c)], np.float32)
    for i, c2 in enumerate(batch_c):
        ious[i] = c1.distance(c2)
    return ious


def shapely_calc_iou_with_contours_1toN(c1: Polygon, batch_c: Iterable[Polygon]):
    '''
    求一个轮廓和一组轮廓的IOU分数，原型
    :param c1:
    :param batch_c:
    :return:
    '''
    ious = np.zeros([len(batch_c)], np.float32)
    for i, c2 in enumerate(batch_c):
        ious[i] = shapely_calc_iou_with_two_contours(c1, c2)
    return ious


def calc_iou_with_contours_1toN(c1, batch_c):
    '''
    求一个轮廓和一组轮廓的IOU分数，包装
    :param c1:
    :param batch_c:
    :return:
    '''
    c1 = tr_my_to_polygon([c1])[0]
    batch_c = tr_my_to_polygon(batch_c)
    ious = shapely_calc_iou_with_contours_1toN(c1, batch_c)
    return ious


def calc_iou_with_contours_NtoM(contours1, contours2):
    '''
    求两组轮廓间两两匹配的IOU分数
    :param contours1:
    :param contours2:
    :return:
    '''
    contours1 = tr_my_to_polygon(contours1)
    contours2 = tr_my_to_polygon(contours2)

    pair_ious = np.zeros([len(contours1), len(contours2)], np.float32)
    for i in range(len(contours1)):
        pair_ious[i] = shapely_calc_iou_with_contours_1toN(contours1[i], contours2)
    return pair_ious


def shapely_calc_occupancy_ratio(contour1, contour2):
    '''
    计算轮廓1和轮廓2的相交区域与轮廓2的占比，原型
    :param contour1: polygon多边形1
    :param contour2: polygon多边形1
    :return:    IOU分数
    '''
    c1 = contour1
    c2 = contour2
    if not c1.intersects(c2):
        return 0.
    area2 = c2.area
    inter_area = c1.intersection(c2).area
    ratio = inter_area / max(area2, 1e-8)
    return ratio


def calc_occupancy_ratio(contour1, contour2):
    '''
    计算轮廓1和轮廓2的相交区域与轮廓2的占比
    :param contour1:
    :param contour2:
    :return:
    '''
    contour1, contour2 = tr_my_to_polygon([contour1, contour2])
    ratio = shapely_calc_occupancy_ratio(contour1, contour2)
    return ratio


def calc_occupancy_ratio_1toN(contour1, contours):
    '''
    计算轮廓1与多个轮廓的相交区域与多个轮廓的占比
    :param contour1:
    :param contours:
    :return:
    '''
    contour1 = tr_my_to_polygon([contour1])[0]
    contours = tr_my_to_polygon(contours)
    ratio = np.zeros([len(contours)], np.float32)
    for i, c in enumerate(contours):
        ratio[i] = shapely_calc_occupancy_ratio(contour1, c)
    return ratio


def calc_occupancy_ratio_Nto1(contours, contour1):
    '''
    计算多个轮廓与轮廓1的相交区域与轮廓1的占比
    :param contours:
    :param contour1:
    :return:
    '''
    contour1 = tr_my_to_polygon([contour1])[0]
    contours = tr_my_to_polygon(contours)
    ratio = np.zeros([len(contours)], np.float32)
    for i, c in enumerate(contours):
        ratio[i] = shapely_calc_occupancy_ratio(c, contour1)
    return ratio


def fusion_im_contours(im, contours, classes, class_to_color_map, copy=True):
    '''
    融合原图和一组轮廓到一张图像
    :param im:                  输入图像，要求为 np.array
    :param contours:            输入轮廓
    :param classes:             每个轮廓的类别
    :param class_to_color_map:  每个类别的颜色
    :return:
    '''
    assert set(classes).issubset(set(class_to_color_map.keys()))
    assert len(contours) == len(classes)
    if copy:
        im = im.copy()
    clss = np.asarray(classes)
    for cls in set(clss):
        cs = list_multi_get_with_bool(contours, clss == cls)
        im = draw_contours(im, cs, class_to_color_map[cls])
    return im


def shapely_merge_to_single_contours(polygons: List[Polygon]):
    '''
    合并多个轮廓到一个轮廓，以第一个轮廓为主，不以第一个轮廓相交的其他轮廓将会被忽略，原型
    :param polygons: 多个多边形，其中第一个为需要融合的主体，后面的如果与第一个不相交，则会忽略
    :return: 已合并的轮廓
    '''
    ps = list(polygons)
    # 逆序删除，避免问题
    # range不能逆序，需要先转换为list
    for i in list(range(1, len(ps)))[::-1]:
        if not ps[0].intersects(ps[i]):
            del ps[i]
    p = unary_union(ps)
    # p 可能是单个多边形，也可能是很多个多边形
    ps = shapely_ensure_polygon_list(p)
    # 保留最大面积的
    areas = [c.area for c in ps]
    p = ps[np.argmax(areas)]
    return p


def merge_to_single_contours(contours, auto_simple=True):
    '''
    合并多个轮廓到一个轮廓，以第一个轮廓为主，不以第一个轮廓相交的其他轮廓将会被忽略
    :param contours: 多个轮廓
    :param auto_simple: 是否自动优化生成的轮廓
    :return: 一个轮廓
    '''
    ps = tr_my_to_polygon(contours)
    p = shapely_merge_to_single_contours(ps)
    c = tr_polygons_to_my([p], contours[0].dtype)[0]
    if auto_simple:
        c = simple_contours([c], epsilon=0)[0]
    return c


def shapely_merge_multi_contours(polygons: List[Polygon]):
    '''
    直接合并多个轮廓
    :param polygons: 多个轮廓，要求为shapely的格式
    :return:
    '''
    ps = unary_union(polygons)
    ps = shapely_ensure_polygon_list(ps)
    return ps


def shapely_merge_multi_contours_sort_by_area(polygons: List[Polygon]):
    '''
    直接合并所有相交的轮廓，计算合并后的轮廓与原始轮廓的IOU，取最大IOU的原始轮廓ID
    :param polygons: 多个轮廓，要求为Polygon列表
    :return: 返回合并后的剩余轮廓，和保留轮廓位于原列表时的ID
    '''
    ps = shapely_merge_multi_contours(polygons)
    ids = []
    for p in ps:
        ious = shapely_calc_iou_with_contours_1toN(p, polygons)
        ids.append(np.argmax(ious))
    return ps, ids


def merge_multi_contours_sort_by_area(contours):
    '''
    直接合并所有相交的轮廓，计算合并后的轮廓与原始轮廓的IOU，取最大IOU的原始轮廓ID
    :param contours: 多个轮廓，要求为我的格式
    :return: 返回合并后的剩余轮廓，和保留轮廓位于原列表时的ID
    '''
    polygons = tr_my_to_polygon(contours)
    ps, ids = shapely_merge_multi_contours_sort_by_area(polygons)
    cs = tr_polygons_to_my(ps)
    return cs, ids


def merge_multi_contours(contours):
    '''
    直接合并多个轮廓
    :param contours: 多个轮廓，要求为我的格式
    :return: 返回合并后的剩余轮廓
    '''
    polygons = tr_my_to_polygon(contours)
    mps = shapely_merge_multi_contours(polygons)
    cs = tr_polygons_to_my(mps)
    return cs


def shapely_inter_contours_1toN(c1: Polygon, batch_c: List[Polygon]):
    '''
    计算一个轮廓与轮廓组的相交轮廓
    :param c1:
    :param batch_c:
    :return:
    '''
    cs = []
    for c in batch_c:
        if c1.intersects(c):
            inter = c1.intersection(c)
            inter = shapely_ensure_polygon_list(inter)
            # 排除inter不是多边形或多个多边形的情况
            cs.extend(inter)
    cs = shapely_merge_multi_contours(cs)
    return cs


def inter_contours_1toN(c1, batch_c):
    '''
    计算一个轮廓与轮廓组的相交轮廓
    :param c1:
    :param batch_c:
    :return:
    '''
    c1 = tr_my_to_polygon([c1])[0]
    batch_c = tr_my_to_polygon(batch_c)
    cs = shapely_inter_contours_1toN(c1, batch_c)
    cs = tr_polygons_to_my(cs)
    return cs


def shapely_morphology_contour(contour: Polygon, distance, resolution=16):
    '''
    对轮廓进行形态学操作，例如膨胀和腐蚀
    对轮廓腐蚀操作后，可能会返回多个轮廓，也可能返回一个空轮廓，此时做好检查，并排除
    因为可能返回多个轮廓，所以统一用list来包装
    :param contour:                 输入轮廓
    :param distance:                形态学操作距离
    :param resolution:              分辨率
    :return:
    '''
    out_c = contour.buffer(distance=distance, resolution=resolution)
    out_c = shapely_ensure_polygon_list(out_c)
    out_c = [c for c in out_c if not c.is_empty and c.is_valid and c.area > 0]
    return out_c


def morphology_contour(contour: np.ndarray, distance, resolution=16):
    '''
    对轮廓进行形态学操作，例如膨胀和腐蚀
    对轮廓腐蚀操作后，可能会返回多个轮廓
    因为可能返回多个轮廓，所以统一用list来包装
    :param contour:                 输入轮廓
    :param distance:                形态学操作距离
    :param resolution:              分辨率
    :return:
    '''
    c = tr_my_to_polygon([contour])[0]
    out_cs = shapely_morphology_contour(c, distance, resolution)
    out_cs = tr_polygons_to_my(out_cs, contour.dtype)
    return out_cs


def shapely_is_valid_contours(polygons: List[Polygon]):
    '''
    检查每个轮廓是否是有效的
    :param contours:            Polygon列表
    :return: bools              布尔列表
    '''
    bs = []
    for p in polygons:
        if not p.is_empty and p.is_valid and p.area > 0:
            bs.append(True)
        else:
            bs.append(False)
    return bs


def is_valid_contours(contours):
    '''
    检查每个轮廓是否是有效的
    :param contours:            轮廓列表
    :return: bools              布尔列表
    '''
    bs = []
    for c in contours:
        if len(c) < 3 or calc_contour_area(c) <= 0:
            bs.append(False)
        else:
            bs.append(True)
    return bs


def is_contours_too_close_border(contours, hw, factor):
    '''
    检测轮廓是否过于靠近边界
    :param contours:    轮廓
    :param hw:          窗口大小
    :param factor:      靠近百分比，值域[0, 1]，常用0.1
    :return:
    '''
    factor = factor / 2
    ids = []
    tlbr = np.asarray([hw[0]*factor, hw[1]*factor, hw[0]*(1-factor), hw[1]*(1-factor)])
    for i, c in enumerate(contours):
        if np.any(c<tlbr[None, :2]) or np.any(c>tlbr[None, 2:]):
            ids.append(i)
    return ids


def check_points_in_contour(points, contour):
    '''
    检查点集是否在轮廓内，大于0代表在轮廓内部，等于0代表在轮廓边上，小于0代表在多边形外部
    :param points:  点列表，[N, yx]
    :param contour: 单个轮廓，我的格式
    :return: bools
    '''
    points = np.float32(points)
    bs = np.zeros(len(points), np.int8)
    c = tr_my_to_cv_contours([contour])[0]
    for i, pt in enumerate(points[:, ::-1]):
        bs[i] = cv2.pointPolygonTest(c, pt, False)
    return bs


def check_point_in_contours(point, contours):
    '''
    检查点集是否在轮廓内，大于0代表在轮廓内部，等于0代表在轮廓边上，小于0代表在多边形外部
    :param points:  单个点，[yx]
    :param contour: 轮廓列表，我的格式
    :return: bools
    '''
    point = np.float32(point)
    bs = np.zeros(len(contours), np.int8)
    cs = tr_my_to_cv_contours(contours)
    pt = point[::-1]
    for i, c in enumerate(cs):
        bs[i] = cv2.pointPolygonTest(c, pt, False)
    return bs


def check_point_in_contour(point, contour):
    '''
    检查点集是否在轮廓内，大于0代表在轮廓内部，等于0代表在轮廓边上，小于0代表在多边形外部
    :param point:  单个点，[yx]
    :param contour: 单个轮廓，我的格式
    :return: bool
    '''
    point = np.float32(point)
    c = tr_my_to_cv_contours([contour])[0]
    pt = point[::-1]
    return cv2.pointPolygonTest(c, pt, False)


def sort_contours_by_area(conts, reverse=False):
    '''
    按面积排序轮廓，默认从小到大
    :param conts:
    :param reverse: 是否反转为从大到小排列
    :return:
    '''
    areas = calc_contours_area(conts)
    ids = np.argsort(areas)
    if reverse:
        ids = ids[::-1]
    cs = list_multi_get_with_ids(conts, ids)
    return cs


def calc_dice_contours(contours1, contours2, *, return_area=False):
    '''
    计算两个轮廓组间的Dice
    :param contours1: 轮廓组，我的格式
    :param contours2: 轮廓组，我的格式
    :return:
    '''
    overlap_conts = []
    for c in contours1:
        overlap_conts.extend(inter_contours_1toN(c, contours2))
    overlap_area = np.sum([calc_contour_area(c) for c in overlap_conts])
    pred_area = np.sum([calc_contour_area(c) for c in contours1])
    label_area = np.sum([calc_contour_area(c) for c in contours2])
    dice = 2 * overlap_area / max(pred_area + label_area, 1e-8)

    if return_area:
        return dice, overlap_area, pred_area, label_area
    else:
        return dice


def apply_affine_to_contours(contours, M):
    '''
    对多个轮廓进行仿射变换
    :param contours: 要求输入格式为 [[yx, yx, ...], [yx, yx, ...], ...]
    :param M: 3x3 或 2x3 变换矩阵
    :return:
    '''
    new_conts = []
    for cont in contours:
        new_conts.append(point_tool.apply_affine_to_points(cont, M))
    return new_conts


def get_contours_shortest_link_pair(contours1, contours2, iou_th: float, *, pair_ious=None):
    '''
    求两组轮廓之间最大IOU配对
    :param contours1:   轮廓组1
    :param contours2:   轮廓组2
    :param iou_th:      iou阈值
    :param pair_ious:   如果已有现成的配对IOU表，则使用现成的，可以省去生成IOU表的时间。IOU表 可用 calc_iou_with_contours_NtoM 函数生成
    :return: 返回轮廓的编号和他们的IOU
    '''

    if pair_ious is None:
        pair_ious = calc_iou_with_contours_NtoM(contours1, contours2)
    else:
        assert pair_ious.shape == (len(contours1), len(contours2)), 'Error! Bad param pair_ious.'

    out_contact_ids1, out_contact_ids2, out_contact_iou = universal_get_shortest_link_pair(pair_ious, iou_th, np.greater)
    return out_contact_ids1, out_contact_ids2, out_contact_iou
    

if __name__ == '__main__':
    c1 = np.array([[0, 0], [0, 100], [100, 100], [100, 0]], np.int)
    c2 = c1.copy()
    c2[:, 1] += 50
    print(calc_contour_area(c1))
    print(calc_iou_with_two_contours(c1, c2))
    print(calc_iou_with_contours_1toN(c1, [c1,c2]))
    print(merge_to_single_contours([c1, c2]))
    im = np.zeros([512, 512, 3], np.uint8)
    im2 = draw_contours(im, [c1, c2], [0, 0, 255], 2)
    im3 = fusion_im_contours(im, [c1, c2], [0, 1], {0: [255, 0, 0], 1: [0, 255, 0]})
    cv2.imshow('show', im2[..., ::-1])
    cv2.imshow('show2', im3[..., ::-1])
    cv2.waitKey(0)

    c3 = np.array([[0, 0], [0, 1]], np.int)
    tr_my_to_polygon([c3])

    from affine_matrix_tool import *
    M = make_rotate(45, (50, 50), None)
    cont = np.array([[25, 10], [25, 80], [75, 75], [75, 25]], np.float32)
    im = np.zeros([100, 100])
    im = draw_contours(im, [cont], 255, 2)
    cv2.imshow('asd', im)
    cv2.waitKey(0)
    cont = apply_affine_to_contours([cont], M)[0]
    im = draw_contours(im, [cont], 255, 2)
    cv2.imshow('asd2', im)
    cv2.waitKey(0)
