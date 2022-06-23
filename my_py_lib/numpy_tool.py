import numpy as np
from scipy.special import softmax
from scipy.special import expit as sigmoid


def one_hot(class_array: np.ndarray, class_num, dim = -1, dtype = np.int32):
    '''
    可将[D1, D2, D3, ..., DN] 矩阵转换为 [D1, D2, ..., DN, D(N+1)] 的独热矩阵
    :param class_array: [D1, D2, ..., DN] 类别矩阵
    :param class_num:   类别数量
    :return:
    '''
    class_array = np.array(class_array, dtype=np.int32, copy=False)
    new_shape = [*([1]*class_array.ndim), class_num]
    a = np.arange(class_num).reshape(new_shape)
    b = np.equal(class_array[..., None], a).astype(dtype)
    if dim != -1:
        b = np.moveaxis(b, -1, dim)
    return b


def one_hot_invert(onehot_array, dim=-1):
    '''
    上面one_hot的逆操作
    可将[D1, D2, D3, ..., DN] 的独热矩阵转换为 [D1, D2, ..., D(N-1)] 的类别矩阵
    :param onehot_array: [D1, D2, ..., DN] 独热矩阵
    :param dim: 操作的维度，默认为-1
    :return: y => class array
    '''
    class_arr = np.argmax(onehot_array, axis=dim)
    return class_arr


# def softmax(x, axis=-1):
#     return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


def tr_figure_to_array(fig):
    '''
    转换 matplotlib 的 figure 到 numpy 数组
    :param fig:
    '''
    fig.canvas.draw()
    mv = fig.canvas.buffer_rgba()
    im = np.asarray(mv)
    # 原图是 rgba，下面去除透明通道
    im = im[..., :3]
    # 需要复制，否则会映射到一个matlibplot的重用内存区，导致返回的图像会被破坏
    im = im.copy()
    return im


def round_int(x, dtype=np.int32):
    '''
    把一个数字变成整数
    :param x:
    :return:
    '''
    return np.asarray(np.round(x), dtype=dtype)


round_int32 = round_int


def ceil_int(x, dtype=np.int32):
    '''
    把一个数字变成整数
    :param x:
    :return:
    '''
    return np.asarray(np.ceil(x), dtype=dtype)


def universal_get_shortest_link_pair(dist_table: np.ndarray, dist_th, compare_op=np.less):
    '''
    通用函数，根据距离表得到最短链匹配ID
    :param dist_table:  距离表，要求维度数量等于2
    :param dist_th:     距离阈值，在距离阈值以外的区域不参与匹配
    :param compare_op:  比较操作，默认为 np.less，代表当 dists < dist_th 时为真。
    :return:
    '''
    assert dist_table.ndim == 2
    ids1 = np.arange(dist_table.shape[0])
    ids2 = np.arange(dist_table.shape[1])

    ids1 = np.tile(ids1[:, None], [1, dist_table.shape[1]]).reshape([-1])
    ids2 = np.tile(ids2[None, :], [dist_table.shape[0], 1]).reshape([-1])
    ids_pair = np.stack([ids1, ids2], 1)
    del ids1, ids2

    dists = dist_table.reshape([-1])

    bs = compare_op(dists, dist_th)
    select_dists = dists[bs]
    select_ids_pair = ids_pair[bs]

    sort_ids = np.argsort(select_dists)[::-1]
    select_dists = select_dists[sort_ids]
    select_ids_pair = select_ids_pair[sort_ids]

    out_contact_ids1 = []
    out_contact_ids2 = []
    out_contact_dists = []

    for dist, (p1, p2) in zip(select_dists, select_ids_pair):
        if p1 not in out_contact_ids1 and p2 not in out_contact_ids2:
            out_contact_ids1.append(p1)
            out_contact_ids2.append(p2)
            out_contact_dists.append(dist)

    return out_contact_ids1, out_contact_ids2, out_contact_dists


if __name__ == '__main__':
    class_num = 10
    a = np.random.randint(0, class_num, (3, 128, 128))
    h = one_hot(class_array=a, class_num=class_num)
    assert np.all(a == np.argmax(h, -1))

    import matplotlib.pyplot as plt
    import cv2

    figure = plt.figure()
    plot = figure.add_subplot(111)
    x = np.arange(1, 100, 0.1)
    y = np.sin(x) / x
    plot.plot(x, y)

    image = tr_figure_to_array(figure)
    cv2.imshow("tr_figure_to_array", image)
    cv2.waitKey(0)
