'''
使用matplotlib来绘图的工具库
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
try:
    from .numpy_tool import tr_figure_to_array
except ImportError:
    from numpy_tool import tr_figure_to_array


def process_param_pts_2d(pts):
    '''
    通用处理2D点输入参数
    :param pts: 点坐标输入，要求形状为 [N, xy]，可以是ndarray,tuple,list
    :return: 返回为 np.ndarray [N,2],float32
    '''
    if not isinstance(pts, np.ndarray):
        pts = np.asarray(pts, np.float32).reshape([-1, 2])

    if pts.dtype != np.float32:
        # 如果不是float32的浮点数，则转换为float32浮点数
        pts = np.float32(pts)

    return pts


def process_param_classes(classes):
    '''
    通用处理类别号输入参数
    :param classes: 类别号输入，要求形状为 [N,]，可以是ndarray,tuple,list
    :return: 返回为 np.ndarray [N,],int32
    '''
    if classes is None:
        classes = np.zeros([len(pts)], np.int32)
    if not isinstance(classes, np.ndarray):
        classes = np.asarray(classes, np.int32).reshape([-1])
    if np.issubdtype(classes.dtype, np.integer) and classes.dtype != np.int32:
        classes = np.asarray(classes, np.int32)

    return classes


def process_param_colors_rgb(colors):
    '''
    通用处理颜色输入参数
    :param colors: 颜色输入，要求形状为 [N,3]，可以是ndarray,tuple,list，如果输入类型是整数，将会除255.，如果输入是浮点数，将保持原样
    :return: 返回为 np.ndarray [N,3],float32
    '''
    if colors is None:
        # 如果没有输入颜色，则默认黑色
        colors = np.zeros([1, 3], np.float32)
    if not isinstance(colors, np.ndarray):
        # 如果输入不是numpy数组，则转换为numpy数组
        colors = np.asarray(colors).reshape([-1, 3])
        # 要求类型为 整数类或浮点类
        assert np.issubdtype(colors.dtype, np.integer) or np.issubdtype(colors.dtype, np.floating)

    if np.issubdtype(colors.dtype, np.integer):
        # 如果是整数类型，则除于255.得到0-1的浮点颜色值
        colors = np.asarray(colors, np.float32) / 255.

    if colors.dtype != np.float32:
        # 如果不是float32的浮点数，则转换为float32浮点数
        colors = np.float32(colors)
    return colors


def plot_scatter_2d(pts, *, classes=None, class_colors=None, class_names=None, name: str = 'scatter',
                    fig=None, ax=None, subplot=111, figsize=(8.00, 8.00), dpi=100, title=None,
                    xlabel=None, ylabel=None, legend=False, tight_layout=False,
                    return_pic=False):
    '''
    绘制散点图
    :param pts:         点坐标输入，要求形状为 [N, xy]，可以是ndarray,tuple,list
    :param classes:     点的类别号
    :param class_colors:每个类别号的颜色
    :param class_names: 每个类别号类别的名字
    :param name:        图的名字
    :param fig:         若不为None，则使用输入的fig
    :param ax:          若不为None，则使用输入的ax
    :param subplot:     若ax为None，则在fig的subplot指定位置新建ax
    :param figsize:     若fig为None，则指定新建fig的figsize
    :param dpi:         若fig为None，则指定新建fig的dpi
    :param title:       若为True，标题则使用name，若为str，则使用输入，若为其他，则忽略
    :param xlabel:      若为str，则绘制x坐标轴名字，若为其他，则忽略
    :param ylabel:      若为str，则绘制y坐标轴名字，若为其他，则忽略
    :param legend:      是否绘制图例
    :param tight_layout:是否自动优化布局
    :param return_pic:  是否额外返回绘制好的图像数组
    :return:
    '''

    # 处理点坐标
    pts = process_param_pts_2d(pts)

    # 处理类别号
    classes = process_param_classes(classes)

    # 处理颜色
    class_colors = process_param_colors_rgb(class_colors)

    # 是否只有一种颜色
    one_color = class_colors.shape[0] == 1

    # 获得类别集合
    uni_classes = np.unique(classes)

    assert pts.ndim == 2 and pts.shape[1] == 2 and pts.dtype == np.float32
    assert classes.ndim == 1 and classes.dtype == np.int32
    assert len(pts) == len(classes)
    assert class_colors.ndim == 2 and class_colors.shape[1] == 3 and class_colors.dtype == np.float32
    assert class_colors.shape[0] == 1 or class_colors.shape[0] >= np.max(uni_classes)

    # 是否新建画布
    if fig is None:
        fig = plt.figure(name, figsize=figsize, dpi=dpi, clear=True)

    # 是否新建一个绘图坐标系
    if ax is None:
        ax = fig.add_subplot(subplot)

    # 检查标题
    if title is True:
        ax.set_title(name)
    elif isinstance(title, str):
        ax.set_title(title)

    # 检查标签
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # 对点分组
    groups_classes_pts = {}
    for c in uni_classes:
        groups_classes_pts[c] = pts[classes == c]

    # 开始绘图
    for cls, cls_pts in groups_classes_pts.items():
        color = class_colors[0] if one_color else class_colors[cls]
        label = cls if class_names is None else class_names[cls]
        a = ax.scatter(cls_pts[:, 0], cls_pts[:, 1], color=color, label=label)

    # 检查图例
    if legend is not None:
        if legend is True:
            # ax.legend(loc='upper right')  # 图例放在左上角，图内
            # ax.legend(loc=(1, 0))         # 图例从右下角开始，图外
            ax.legend(loc='upper left', bbox_to_anchor=(1.01,1), borderaxespad=0)   # 图例从右上角开始，图外
        elif isinstance(legend, dict):
            ax.legend(**legend)

    # 是否使用自动调整布局
    if tight_layout:
        fig.tight_layout()

    r = (fig, ax)
    # 是否返回图像
    if return_pic:
        r = r + (tr_figure_to_array(fig),)

    return r


if __name__ == '__main__':
    pts = [[10, 10], [20, 20], [30, 30], [0, 10], [20, 0]]
    cls = [0, 0, 1, 1, 0]
    cls_colors = [[0, 128, 255], [128, 0, 128]]
    cls_names = ['a', 'b']
    _, _, im = plot_scatter_2d(pts, classes=cls, class_colors=cls_colors,
                               class_names=cls_names, return_pic=True, title='title',
                               legend=True, xlabel='X', ylabel='Y')
    plt.show()
    cv2.imshow('asd', im[..., ::-1])
    cv2.waitKey()
