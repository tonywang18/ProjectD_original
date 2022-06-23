'''
图像，坐标仿射增强工具
'''


import cv2
import numpy as np
from typing import Union, Tuple, Iterable
try:
    from im_tool import pad_picture
    import coord_tool
    from affine_matrix_tool import make_move, make_rotate, make_scale, make_shear
    import point_tool
except ModuleNotFoundError:
    from .im_tool import pad_picture
    from . import coord_tool
    from .affine_matrix_tool import make_move, make_rotate, make_scale, make_shear
    from . import point_tool


# # 旋转，平移，缩放，切变
#
# def make_rotate(angle=180., center_yx=(0.5, 0.5), img_hw: Union[None, Tuple]=(512, 512)):
#     '''
#     旋转
#     :param angle: 顺时针旋转角度
#     :param center_yx: 如果 img_hw 不为 None，则为百分比坐标，否则为绝对坐标
#     :param img_hw: 图像大小，单位为像素
#     :return:
#     '''
#     if img_hw is not None:
#         center_yx = (img_hw[0] * center_yx[0], img_hw[1] * center_yx[1])
#
#     R = np.eye(3)
#     # 这里加个符号使其顺时针转动
#     R[:2] = cv2.getRotationMatrix2D(angle=-angle, center=center_yx[::-1], scale=1.)
#     return R
#
#
# def make_move(move_yx=(0.2, 0.2), img_hw: Union[None, Tuple]=(512, 512)):
#     '''
#     平移
#     :param move_yx: 平移距离，当 img_hw 不是None时，单位为图像大小百分比，如果 img_hw 是 None，则为像素单位
#     :param img_hw: 图像大小，单位为像素
#     :return:
#     '''
#     move_yx = list(move_yx)
#     if img_hw is not None:
#         move_yx[1] = move_yx[1] * img_hw[1]
#         move_yx[0] = move_yx[0] * img_hw[0]
#
#     T = np.eye(3)
#     T[0, 2] = move_yx[1]
#     T[1, 2] = move_yx[0]
#     return T
#
#
# def make_scale(scale_yx=(2., 2.), center_yx=(0.5, 0.5), img_hw: Union[None, Tuple]=(512, 512)):
#     '''
#     缩放
#     :param scale_yx: 单位必须为相对图像大小的百分比
#     :param center_yx: 变换中心位置，当 img_hw 不是None时，单位为图像大小百分比，如果 img_hw 是 None，则为像素单位
#     :param img_hw: 图像大小，单位为像素
#     :return:
#     '''
#     if img_hw is not None:
#         center_yx = (img_hw[0] * center_yx[0], img_hw[1] * center_yx[1])
#
#     S = np.eye(3)
#     S[0, 0] = scale_yx[1]
#     S[1, 1] = scale_yx[0]
#     S[0, 2] = (1-scale_yx[1]) * center_yx[1]
#     S[1, 2] = (1-scale_yx[0]) * center_yx[0]
#     return S
#
#
# def make_shear(shear_yx=(1.2, 1.2)):
#     '''
#     切变
#     :param shear_yx: 单位为角度；变量1，图像x边与窗口x边角度，变量2，图像y边与窗口y边角度
#     :return:
#     '''
#     # Shear
#     S = np.eye(3)
#     S[0, 1] = np.tan(shear_yx[1] * np.pi / 180)  # x shear (deg)
#     S[1, 0] = np.tan(shear_yx[0] * np.pi / 180)  # y shear (deg)
#     return S


def apply_affine_to_img(img, M, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0), out_hw: tuple=None):
    '''
    :param img: 图像
    :param M: 变换矩阵
    :param interpolation: 平滑方式
    :param borderMode: 填充方式
    :param borderValue: 边缘填充颜色
    :return:
    '''
    if out_hw is None:
        hw = img.shape[:2]
    else:
        hw = tuple(out_hw)
    img = cv2.warpPerspective(img, M, dsize=hw[::-1], flags=interpolation, borderMode=borderMode, borderValue=borderValue)
    return img


# def apply_affine_to_coords(coords_y1x1y2x2, M):
#     '''
#     对坐标进行仿射变换
#     :param coords_y1x1y2x2: 要求输入格式为 [y1x2y2x2, y1x2y2x2, ...]
#     :param M: 3x3 或 2x3 变换矩阵
#     :return:
#     '''
#     coords_x1y1x2y2 = coord_tool.yxhw2xywh(coords_y1x1y2x2)
#     # 对矩阵乘法不熟悉，所以使用易于理解的方式处理
#
#     # 分解矩阵为旋转矩阵和平移矩阵，因为貌似旋转和平移没法在一次运算中完成
#     R = M[:2, 0:2]
#     T = M[:2, 2:3]
#
#     # 先得到包围框的四个坐标点，需要变换后的矩形的外接正矩形
#     coords_x1y1_x2y1_x1y2_x2y2 = coords_x1y1x2y2[:, [0, 1, 2, 1, 0, 3, 2, 3]]
#     xy = coords_x1y1_x2y1_x1y2_x2y2.reshape([-1, 2])
#     xy_t = R @ xy.T
#     xy_t = T + xy_t
#     xy = xy_t.T
#
#     # 变换完了...
#     # coords_x1y1_x2y1_x1y2_x2y2 = xy.reshape([-1, 8])
#     coords_groups_xy = xy.reshape([-1, 4, 2])
#     # 将x坐标分到一组，y坐标分到一组，便于直接求大小
#     coords_xy_groups = np.transpose(coords_groups_xy, [0, 2, 1])
#     coord_xy_left_top = np.min(coords_xy_groups, 2)
#     coord_xy_right_bottom = np.max(coords_xy_groups, 2)
#     coord_x1y1x2y2 = np.concatenate([coord_xy_left_top, coord_xy_right_bottom], 1)
#     out = coord_tool.xywh2yxhw(coord_x1y1x2y2)
#
#     return out


def apply_affine_to_coords(coords_y1x1y2x2, M):
    '''
    对坐标进行仿射变换
    :param coords_y1x1y2x2: 要求输入格式为 [y1x2y2x2, y1x2y2x2, ...]
    :param M: 3x3 或 2x3 变换矩阵
    :return:
    '''
    # 先得到包围框的四个坐标点，需要变换后的矩形的外接正矩形
    coords_y1x1_y1x2_y2x1_y2x2 = coords_y1x1y2x2[:, [0, 1, 0, 3, 2, 1, 2, 3]]
    pts_yx = np.reshape(coords_y1x1_y1x2_y2x1_y2x2, [-1, 2])
    pts_yx = point_tool.apply_affine_to_points(pts_yx, M)
    pts_yx = np.reshape(pts_yx, [-1, 8])

    coords_groups_yx = pts_yx.reshape([-1, 4, 2])
    # 将x坐标分到一组，y坐标分到一组，便于直接求大小
    coords_yx_groups = np.transpose(coords_groups_yx, [0, 2, 1])
    coord_yx_left_top = np.min(coords_yx_groups, 2)
    coord_yx_right_bottom = np.max(coords_yx_groups, 2)
    coord_y1x1y2x2 = np.concatenate([coord_yx_left_top, coord_yx_right_bottom], 1)
    # out = coord_tool.xywh2yxhw(coord_y1x1y2x2)

    return coord_y1x1y2x2


def coords_clip(coords_y1x1y2x2, img_hw):
    new_coords = np.clip(coords_y1x1y2x2, [0, 0, 0, 0], [*img_hw, *img_hw])
    return new_coords


def random_affine_matrix(move_low_high=[-0.3, 0.3], angle_low_high=[-180, 180], scale_low_high=[0.7, 1.3], shear_low_high=[-20, 20], img_hw=[512, 512]):
    move_ratio_h = np.random.uniform(move_low_high[0], move_low_high[1])
    move_ratio_w = np.random.uniform(move_low_high[0], move_low_high[1])
    T = make_move([move_ratio_h, move_ratio_w], img_hw)

    angle = np.random.uniform(angle_low_high[0], angle_low_high[1])
    R = make_rotate(angle, center_yx=[0.5, 0.5], img_hw=img_hw)

    scale_ratio_h = np.random.uniform(scale_low_high[0], scale_low_high[1])
    scale_ratio_w = np.random.uniform(scale_low_high[0], scale_low_high[1])
    S = make_scale([scale_ratio_h, scale_ratio_w], center_yx=[0.5, 0.5], img_hw=img_hw)

    shear_ratio_h = np.random.uniform(shear_low_high[0], shear_low_high[1])
    shear_ratio_w = np.random.uniform(shear_low_high[0], shear_low_high[1])
    Sh = make_shear([shear_ratio_h, shear_ratio_w])

    # 运算方向注意 T->R->S->Sh
    M = Sh @ S @ R @ T
    return M


if __name__ == '__main__':
    import imageio
    import os
    import im_tool

    impath = os.path.dirname(__file__) + '/laska.png'
    im = imageio.imread(impath)
    coord_y1x1y2x2 = np.array([[75, 50, 150, 175]])

    ori_im = im_tool.draw_boxes_and_labels_to_image(im, [0], coord_tool.y1x1y2x2_to_x1y1x2y2(coord_y1x1y2x2))
    im_tool.show_image(ori_im)

    # 测试旋转，注意这里得到的是包围框旋转后的外接矩形，所以宽高比不与原来的一样
    R = make_rotate(60, [0.5, 0.5], im.shape[:2])
    im_r = apply_affine_to_img(im, R, borderValue=[128, 128, 128])
    coord_r = apply_affine_to_coords(coord_y1x1y2x2, R)
    coord_r = coord_tool.y1x1y2x2_to_x1y1x2y2(coord_r)
    im_r = im_tool.draw_boxes_and_labels_to_image(im_r, [0], coord_r)
    im_tool.show_image(im_r)

    # 测试平移
    T = make_move([0.2, 0.5], im.shape[:2])
    im_t = apply_affine_to_img(im, T, borderValue=[128, 128, 128])
    coord_t = apply_affine_to_coords(coord_y1x1y2x2, T)
    coord_t = coord_tool.y1x1y2x2_to_x1y1x2y2(coord_t)
    im_t = im_tool.draw_boxes_and_labels_to_image(im_t, [0], coord_t)
    im_tool.show_image(im_t)

    # 测试缩放
    S = make_scale([1.2, 0.5], [0.5, 0.5], im.shape[:2])
    im_s = apply_affine_to_img(im, S, borderValue=[128, 128, 128])
    coord_s = apply_affine_to_coords(coord_y1x1y2x2, S)
    coord_s = coord_tool.y1x1y2x2_to_x1y1x2y2(coord_s)
    im_s = im_tool.draw_boxes_and_labels_to_image(im_s, [0], coord_s)
    im_tool.show_image(im_s)

    # 测试切变
    SH = make_shear([0, 30])
    im_sh = apply_affine_to_img(im, SH, borderValue=[128, 128, 128])
    coord_sh = apply_affine_to_coords(coord_y1x1y2x2, SH)
    coord_sh = coord_tool.y1x1y2x2_to_x1y1x2y2(coord_sh)
    im_sh = im_tool.draw_boxes_and_labels_to_image(im_sh, [0], coord_sh)
    im_tool.show_image(im_sh)

    # 测试组合
    # 运算方向 T -> R -> S
    M = S @ R @ T
    im_m = apply_affine_to_img(im, M, borderValue=[128, 128, 128])
    coord_m = apply_affine_to_coords(coord_y1x1y2x2, M)
    coord_m = coord_tool.y1x1y2x2_to_x1y1x2y2(coord_m)
    im_m = im_tool.draw_boxes_and_labels_to_image(im_m, [0], coord_m)
    im_tool.show_image(im_m)
