'''
The algorithm comes from the paper
"Enhanced Center Coding for Cell Detection with Convolutional Neural Networks"
https://arxiv.org/abs/1904.08864?context=cs.CV

'''

import numpy as np
from skimage.draw import disk as sk_disk


def draw_repel_code_ori(im: np.ndarray, pts: np.ndarray, r, A=1):
    '''
    :param im: [H, W, 1] The image to be drawn.
    :param pts: [N, yx] Input points. You need to ensure that the each point is unique.
    :param r: The radius of the drawing area.
    :param A: Control the gradual change.
    :return:
    '''
    if not isinstance(pts, np.ndarray):
        pts = np.asarray(pts, np.int)
    assert im.ndim == 3 and im.shape[2] == 1

    wait_check_pixs = []
    for pt in pts:
        rr, cc = sk_disk(pt, r, shape=im.shape[:2])
        pts2 = np.stack([rr, cc], 1)
        pts2 = np.asarray(pts2, np.int32)
        wait_check_pixs.extend(pts2)
    check_pixs = np.unique(wait_check_pixs, axis=0)

    # # 源代码，很慢
    # for pix in check_pixs:
    #     ds = np.linalg.norm(pix[None] - pts, 2, 1)
    #     seq_id = np.argsort(ds)[:2]
    #     # close_2pts = list(pts[seq_id])
    #     close_2pts_ds = list(ds[seq_id])
    #     C = 0
    #     if len(close_2pts_ds) == 2:
    #         D = close_2pts_ds[0] * (1 + close_2pts_ds[0] / (np.clip(close_2pts_ds[1], 1e-8, None)))
    #         if D < r:
    #             C = 1 / (1 + D * A)
    #     elif len(close_2pts_ds) == 1:
    #         D = close_2pts_ds[0]
    #         if D < r:
    #             C = 1 / (1 + D * A)
    #     im[pix[0], pix[1]] = C

    # # 这里为上面的并行加速代码。但还有待优化，内存占用太大了。
    # big_ds = np.linalg.norm(np.asarray(check_pixs[:, None] - pts[None], np.float32), 2, 2)
    # big_seq_id = np.argsort(big_ds, axis=-1)[:, :2]
    # big_close_2pts_ds = np.take_along_axis(big_ds, big_seq_id, 1)
    #
    # if big_close_2pts_ds.shape[1] == 2:
    #     D = big_close_2pts_ds[:, 0] * (1 + big_close_2pts_ds[:, 0] / (np.clip(big_close_2pts_ds[:, 1], 1e-8, None)))
    #     C = np.where(D < r, 1 / (1 + D * A), 0)
    # elif big_close_2pts_ds.shape[1] == 1:
    #     D = big_close_2pts_ds[:, 0]
    #     C = np.where(D < r, 1 / (1 + D * A), 0)
    # else:
    #     return im
    # im[check_pixs[:, 0], check_pixs[:, 1]] = C[:, None]

    # 这里为上面的并行加速代码的分段代码，一次性查全部点，占用内存过于吓人。
    n_each_draw = 5000
    draw_count = int(np.ceil(len(check_pixs) / n_each_draw))
    for i in range(draw_count):
        cur_check_pixs = check_pixs[i*n_each_draw:(i+1)*n_each_draw]

        big_ds = np.linalg.norm(np.asarray(cur_check_pixs[:, None] - pts[None], np.float32), 2, 2)
        big_seq_id = np.argsort(big_ds, axis=-1)[:, :2]
        big_close_2pts_ds = np.take_along_axis(big_ds, big_seq_id, 1)

        if big_close_2pts_ds.shape[1] == 2:
            D = big_close_2pts_ds[:, 0] * (1 + big_close_2pts_ds[:, 0] / (np.clip(big_close_2pts_ds[:, 1], 1e-8, None)))
            C = np.where(D < r, 1 / (1 + D * A), 0)
        elif big_close_2pts_ds.shape[1] == 1:
            D = big_close_2pts_ds[:, 0]
            C = np.where(D < r, 1 / (1 + D * A), 0)
        else:
            # 这里是一次性检查全部点，如果没有，可以认为后面也没有了，可以直接退出
            break
        im[cur_check_pixs[:, 0], cur_check_pixs[:, 1]] = C[:, None]

    return im


def draw_repel_code_fast(im: np.ndarray, pts: np.ndarray, r, A=1):
    '''
    Another faster acceleration method code of the above function, but there are accuracy problems.
    :param im: [H, W, 1] The image to be drawn.
    :param pts: [N, yx] Input points. You need to ensure that the each point is unique.
    :param r: The radius of the drawing area.
    :param A: Control the gradual change.
    :return:
    '''
    assert im.ndim == 3 and im.shape[2] == 1

    for pt in pts:
        ds = np.linalg.norm(pt[None,] - pts, 2, 1)
        # 注意，这个点集已经包含了待查询点自身
        # 该处r*2是为了增加更多的检查点，减少不准确问题
        close_pts = pts[ds <= r * 2]
        rr, cc = sk_disk(pt, r, shape=im.shape[:2])
        pts2 = np.stack([rr, cc], 1)
        pts2 = np.asarray(pts2, np.int32)

        big_ds = np.linalg.norm(np.asarray(pts2[:, None] - close_pts[None], np.float32), 2, 2)
        # big_ds = np.linalg.norm(np.asarray(close_pts[None,] - pts2[:, None], np.float32), 2, 2)
        big_seq_id = np.argsort(big_ds, axis=-1)[:, :2]
        big_close_2pts_ds = np.take_along_axis(big_ds, big_seq_id, 1)

        if big_close_2pts_ds.shape[1] == 2:
            D = big_close_2pts_ds[:, 0] * (1 + big_close_2pts_ds[:, 0] / (np.clip(big_close_2pts_ds[:, 1], 1e-8, None)))
            C = np.where(D < r, 1 / (1 + D * A), 0)
        elif big_close_2pts_ds.shape[1] == 1:
            D = big_close_2pts_ds[:, 0]
            C = np.where(D < r, 1 / (1 + D * A), 0)
        else:
            # return im
            continue
        im[pts2[:, 0], pts2[:, 1]] = C[:, None]

    return im


if __name__ == '__main__':
    import cv2

    im = np.zeros([512, 512, 1], np.float32)
    # 注意输入的pt坐标顺序为yx，而不是xy
    pts = np.random.randint(0, 512, [100, 2], np.int32)

    # 不允许有重复的坐标
    pts = np.unique(pts, axis=0)

    oim1 = draw_repel_code_ori(im, pts, 9)
    oim2 = draw_repel_code_fast(im, pts, 9)

    cv2.imshow('oim1', oim1)
    cv2.imshow('oim2', oim2)

    cv2.waitKey(0)
