import os
import imageio
import cv2
import numpy as np
import scipy.io
import random
import math
from a_config import project_root
from my_py_lib.image_over_scan_wrapper import ImageOverScanWrapper
from my_py_lib.coords_over_scan_gen import n_step_scan_coords_gen
from my_py_lib.list_tool import list_multi_get_with_ids
from my_py_lib.im_tool import check_and_tr_umat, ensure_image_has_3dim
import my_py_lib.color_enhance_tool as color_enhance_tool
from skimage.transform import rotate as sk_rotate
from skimage.draw import disk as sk_disk
from my_py_lib.draw_repel_code_tool import draw_repel_code_fast

draw_repel_code = draw_repel_code_fast

# def find_Nclose_pts(pts, N=1):
#     '''
#     对每个点，找到距离自身最近的其他N个点
#     :param pts:
#     :return:
#     '''
#     pts = np.array(pts)
#     close_pts = []
#     if len(pts) == 0:
#         return close_pts
#
#     for i in range(len(pts)):
#         cur_pt = pts[i:i+1]
#         # 删除自身
#         ds = np.linalg.norm(cur_pt - pts, 2, 1)
#         seq_id = np.argsort(ds)[1:N+1]
#         each_close_2pts = list(pts[seq_id])
#         close_pts.append(each_close_2pts)
#
#     return close_pts


def draw_im_points(ori_im, ori_points, ori_clss, n_cls, draw_rect, pt_radius_max, pt_radius_min, use_repel_code, A=0.3):
    '''
    绘制图像
    :param ori_im:
    :param ori_points:
    :param n_cls:
    :param draw_rect:   y1x1y2x2
    :param pt_radius_max:
    :param pt_radius_min:
    :param use_repel_code:
    :return:
    '''
    ori_points = np.asarray(ori_points)
    ori_clss = np.asarray(ori_clss)
    # 使用 ImageOverScanWrapper 可以允许过采样
    wrap_im = ImageOverScanWrapper(ori_im)
    crop_im = wrap_im.get(draw_rect[:2], draw_rect[2:], 0)
    label = np.zeros([draw_rect[2] - draw_rect[0], draw_rect[3] - draw_rect[1], n_cls], np.float32)
    if len(ori_points) == 0:
        # 没有点需要绘制则直接返回
        return crop_im, label
    # 选出在绘画范围内的点
    is_in =  np.all([np.all(ori_points <= draw_rect[None, 2:], 1), np.all(ori_points >= draw_rect[None, :2], 1)], 0)
    in_pts = ori_points[is_in]
    if len(in_pts) == 0:
        return crop_im, label

    # 平移选出的点
    in_pts = in_pts - draw_rect[None, :2]
    in_clss = ori_clss[is_in]

    if not use_repel_code:
        # 原始画法
        for pt, cls in zip(in_pts, in_clss):
            radius = random.uniform(pt_radius_min, pt_radius_max) if pt_radius_max > pt_radius_min else pt_radius_min
            radius = int(np.round(radius))
            rr, cc = sk_disk(pt, radius, shape=label.shape[:2])
            label[rr, cc, cls] = 1
    else:
        # 新的画法，排斥编码
        unique_cls = list(set(in_clss))
        for cls in unique_cls:
            select_pts = in_pts[in_clss == cls]
            draw_repel_code(label[:, :, cls:cls+1], select_pts, pt_radius_max, A=A)

    return crop_im, label


class DatasetReader:
    class_names = ('detection', 'others', 'epithelial', 'fibroblast', 'inflammatory')
    class_ids = (0, 1, 2, 3, 4)
    class_num = len(class_ids)
    assert len(class_names) == len(class_ids)

    def __init__(self, dataset_path=os.path.join(project_root, 'data_2'), data_type='train', pt_radius_min=3, pt_radius_max=6, use_repel_code=False, A=0.3):
        self.data_path = os.path.join(dataset_path, data_type)
        self.pt_radius_min = pt_radius_min
        self.pt_radius_max = pt_radius_max
        self.use_repel_code = use_repel_code
        self.A = A

        assert self.pt_radius_max >= self.pt_radius_min

        self._im_path_list = []
        self._label_pos_list = []
        self._label_cls_list = []
        self._build()

    def _build(self):
        dirs = os.listdir(self.data_path)
        for name in dirs:
            im_dir_path = os.path.join(self.data_path, name)
            if not os.path.isdir(im_dir_path):
                print('Found {} is not a dir'.format(im_dir_path))
                continue

            im_path = os.path.join(im_dir_path, '{}.bmp'.format(name))
            label_pos = []
            label_cls = []

            for cla_name, cla_id in zip(self.class_names, self.class_ids):
                label_path = os.path.join(im_dir_path, '{}_{}.mat'.format(name, cla_name))
                m = scipy.io.loadmat(label_path)['detection']
                assert m.ndim == 2 and m.shape[1] == 2
                m = m.astype(np.int)
                label_pos.extend(m[:, ::-1])            # 坐标格式为yx
                label_cls.extend([cla_id] * len(m))     # 类别

            label_pos = np.asarray(label_pos, np.int32)
            label_cls = np.asarray(label_cls, np.int32)
            self._im_path_list.append(im_path)
            self._label_pos_list.append(label_pos)
            self._label_cls_list.append(label_cls)

        # 最终检查
        assert len(self._im_path_list) == len(self._label_pos_list) == len(self._label_cls_list)

    def __len__(self):
        return len(self._im_path_list)

    # def get_item(self, item_id, use_enhance=False, window_hw=None, *, return_im_info=False):
    #     window_hw = np.array(window_hw, np.int) if window_hw is not None else window_hw
    #     im_path = self._im_path_list[item_id]
    #     label_pos = self._label_pos_list[item_id]
    #     label_cls = self._label_cls_list[item_id]
    #
    #     im = imageio.imread(im_path)
    #     assert im.dtype == np.uint8
    #
    #     more_info = {
    #         'im_path': im_path,
    #         'label_pos': label_pos,
    #         'label_cls': label_cls,
    #     }
    #
    #     if window_hw is not None:
    #         center_yx = np.random.randint(window_hw // 2, (im.shape[0], im.shape[1]) - window_hw // 2, 2, np.int32)
    #         half_hw = window_hw // 2
    #         yx_start = center_yx - half_hw
    #         yx_end = center_yx + half_hw
    #         im, label = draw_im_points(im, label_pos, label_cls, self.class_num, np.concatenate([yx_start, yx_end], 0),
    #                                    self.pt_radius_max, self.pt_radius_min, self.use_repel_code)
    #     else:
    #         im, label = draw_im_points(im, label_pos, label_cls, self.class_num, np.array([0, 0, im.shape[0], im.shape[1]]),
    #                                    self.pt_radius_max, self.pt_radius_min, self.use_repel_code)
    #
    #     if use_enhance:
    #         # rot_code = np.random.choice([-1, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180])
    #         rot_code = np.random.choice([0, 90, -90, 180])
    #         if rot_code != -1:
    #             # cv2.rotate(im, rot_code, im)
    #             im = sk_rotate(im, rot_code, preserve_range=True, order=0).astype(np.uint8)
    #             label = sk_rotate(label, rot_code, preserve_range=True, order=0).astype(np.float32)
    #             # cv2.rotate(label, rot_code, label)
    #         # if np.random.uniform() > 0.7:
    #         #     # 随机高斯模糊
    #         #     ker_sz = 3 # np.random.choice([3, 5])
    #         #     o_im = cv2.GaussianBlur(o_im, (ker_sz, ker_sz), 0)
    #         if np.random.uniform() > 0.5:
    #             # 随机色调调整
    #             im = color_enhance_tool.random_adjust_HSV(im, 0.2, 0.2, 0.1)
    #         if np.random.uniform() > 0.5:
    #             # 随机噪音
    #             im = color_enhance_tool.random_gasuss_noise(im, 0, 0.05)
    #
    #     im = ensure_image_has_3dim(im)
    #     label = ensure_image_has_3dim(label)
    #
    #     if return_im_info:
    #         return im, label, more_info
    #     else:
    #         return im, label

    def get_item(self, item_id, use_enhance=False, window_hw=None):
        window_hw = np.array(window_hw, np.int) if window_hw is not None else window_hw
        im_path = self._im_path_list[item_id]
        label_pos = self._label_pos_list[item_id]
        label_cls = self._label_cls_list[item_id]

        im = imageio.imread(im_path)
        assert im.dtype == np.uint8

        label_det_pts, label_det_pts_without_ignore, group_label_cla_pts, group_label_cla_pts_without_ignore, ignore_pts =\
            self.split_label_pts(label_pos, label_cls, self.pt_radius_max)

        info = {
            'im_path': im_path,
            'label_pos': label_pos,
            'label_cls': label_cls,
            'label_det_pts': label_det_pts,
            'label_det_pts_without_ignore': label_det_pts_without_ignore,
            'group_label_cla_pts': group_label_cla_pts,
            'group_label_cla_pts_without_ignore': group_label_cla_pts_without_ignore,
            'ignore_pts': ignore_pts,
        }

        if window_hw is not None:
            center_yx = np.random.randint(window_hw // 2, (im.shape[0], im.shape[1]) - window_hw // 2, 2, np.int32)
            half_hw = window_hw // 2
            yx_start = center_yx - half_hw
            yx_end = center_yx + half_hw
            rect = np.concatenate([yx_start, yx_end], 0)
        else:
            rect = np.array([0, 0, im.shape[0], im.shape[1]])

        oim, label = draw_im_points(im, label_pos, label_cls, self.class_num, rect, self.pt_radius_max, self.pt_radius_min, self.use_repel_code, A=self.A)
        _, ignore_mask = draw_im_points(im, ignore_pts, [0]*len(ignore_pts), 1, rect, self.pt_radius_max, self.pt_radius_max, False)

        ignore_mask = np.asarray(ignore_mask, np.uint8)

        if use_enhance:
            rot_code = np.random.choice([0, 90, -90, 180])
            if rot_code != -1:
                oim = sk_rotate(oim, rot_code, preserve_range=True, order=0).astype(np.uint8)
                label = sk_rotate(label, rot_code, preserve_range=True, order=0).astype(np.float32)
                ignore_mask = sk_rotate(ignore_mask, rot_code, preserve_range=True, order=0).astype(np.uint8)
            # if np.random.uniform() > 0.7:
            #     # 随机高斯模糊
            #     ker_sz = 3 # np.random.choice([3, 5])
            #     o_im = cv2.GaussianBlur(o_im, (ker_sz, ker_sz), 0)
            if np.random.uniform() > 0.5:
                # 随机色调调整
                oim = color_enhance_tool.random_adjust_HSV(oim, 0.2, 0.2, 0.1)
            if np.random.uniform() > 0.5:
                # 随机噪音
                oim = color_enhance_tool.random_gasuss_noise(oim, 0, 0.05)

        oim = ensure_image_has_3dim(oim)
        label = ensure_image_has_3dim(label)
        ignore_mask = ensure_image_has_3dim(ignore_mask)

        return oim, label, ignore_mask, info

    def get_batch(self, batch_size, use_enhance=False, window_hw=None):
        ids = np.random.randint(0, len(self._im_path_list), batch_size)
        batch_im = []
        batch_label = []
        batch_ignore_mask = []
        batch_info = []
        for i in ids:
            im, label, ignore_mask, info = self.get_item(i, use_enhance, window_hw)
            batch_im.append(im)
            batch_label.append(label)
            batch_ignore_mask.append(ignore_mask)
            batch_info.append(info)
        return np.asarray(batch_im), np.asarray(batch_label), np.asarray(batch_ignore_mask), batch_info

    def get_train_batch_gen(self, batch_count, batch_size, use_enhance=False, window_hw=(256, 256)):
        for i in range(batch_count):
            batch_im, batch_label, batch_ignore_mask, batch_info = self.get_batch(batch_size, use_enhance=use_enhance, window_hw=window_hw)
            yield batch_im, batch_label, batch_ignore_mask, batch_info

    def get_eval_batch_gen(self, batch_size, window_hw=None):
        batch_im = []
        batch_label = []
        batch_ignore_mask = []
        batch_info = []
        for i in range(len(self._im_path_list)):
            # 获取标签原图
            im, label, ignore_mask, info = self.get_item(i, use_enhance=False, window_hw=None)
            wim = ImageOverScanWrapper(im)
            wlabel = ImageOverScanWrapper(label)
            wignore_mask = ImageOverScanWrapper(ignore_mask)
            # 半步长采样
            coords_g = n_step_scan_coords_gen(im.shape[:2], window_hw, 1)
            for yx_start, yx_end in coords_g:
                patch_im = wim.get(yx_start, yx_end)
                patch_label = wlabel.get(yx_start, yx_end)
                patch_ignore_mask = wignore_mask.get(yx_start, yx_end)

                patch_im = ensure_image_has_3dim(patch_im)
                patch_label = ensure_image_has_3dim(patch_label)
                patch_ignore_mask = ensure_image_has_3dim(patch_ignore_mask)

                batch_im.append(patch_im)
                batch_label.append(patch_label)
                batch_ignore_mask.append(patch_ignore_mask)
                batch_info.append(info)

                if len(batch_im) >= batch_size:
                    yield np.asarray(batch_im), np.asarray(batch_label), np.asarray(batch_ignore_mask), batch_info
                    batch_im = []
                    batch_label = []
                    batch_ignore_mask = []
                    batch_info = []

        if len(batch_im) > 0:
            yield np.asarray(batch_im), np.asarray(batch_label), np.asarray(batch_ignore_mask), batch_info

    def split_label_pts(self, label_pos, label_cls, thresh=6):
        '''
        从原标签中分离出检测点和分类点，并且分离出忽略点。
        :param label_det:
        :param label_cla:
        :return:
        '''
        assert len(label_pos) == len(label_cls)

        label_det_pts = []
        label_det_pts_without_ignore = []
        group_label_cla_pts = dict([(k, []) for k in self.class_ids[1:]])
        group_label_cla_pts_without_ignore = dict([(k, []) for k in self.class_ids[1:]])
        ignore_pts = []

        for pos, cla in zip(label_pos, label_cls):
            if cla == 0:
                label_det_pts.append(pos)
            elif cla in self.class_ids[1:]:
                group_label_cla_pts[cla].append(pos)
            else:
                raise AssertionError()

        label_det_pts = np.asarray(label_det_pts)
        label_det_pts_b = np.zeros([len(label_det_pts)], np.bool)   # 检查是否有配对点

        for k in group_label_cla_pts:
            group_label_cla_pts[k] = np.asarray(group_label_cla_pts[k])

        for gid, cur_label_cla_pts in group_label_cla_pts.items():
            for i, pt in enumerate(cur_label_cla_pts):
                if len(label_det_pts) > 0:
                    bs = np.linalg.norm(pt[None] - label_det_pts, 2, axis=1) <= thresh
                    np.logical_or(label_det_pts_b, bs, out=label_det_pts_b)
                    if np.any(bs):
                        group_label_cla_pts_without_ignore[gid].append(pt)
                    else:
                        ignore_pts.append(pt)

        if len(label_det_pts) > 0:
            label_det_pts_without_ignore = label_det_pts[label_det_pts_b]
            ignore_pts.extend(label_det_pts[np.logical_not(label_det_pts_b)])

        return label_det_pts, label_det_pts_without_ignore, group_label_cla_pts, group_label_cla_pts_without_ignore, ignore_pts


if __name__ == '__main__':
    from my_py_lib import im_tool
    from my_py_lib import preload_generator

    ds = DatasetReader(data_type='train', pt_radius_min=9, pt_radius_max=9, use_repel_code=True)

    # test get_item
    for i in range(len(ds)):
        im, label, ignore_mask, info = ds.get_item(i, use_enhance=False)
        cv2.imshow('ori_im', im[..., ::-1])
        cv2.imshow('label_det', label[:, :, 0])
        label = np.transpose(label, [2, 0, 1]).__mul__(255).astype(np.uint8)
        label_tile1 = im_tool.draw_multi_img_in_big_img(label, 1, [600, 1200], [2, 3], pad_color=64)
        cv2.imshow('label_tile1', label_tile1)
        label_tile2 = im_tool.draw_multi_img_in_big_img(label * (label > 128), 1, [600, 1200], [2, 3], pad_color=64)
        cv2.imshow('label_tile2', label_tile2)
        cv2.imshow('ignore_mask', cv2.resize(ignore_mask * 255, (300, 300), interpolation=cv2.INTER_AREA))
        # os.makedirs('tmp', exist_ok=True)
        # basename = os.path.splitext(os.path.basename(info['im_path']))[0]
        # imageio.imwrite('tmp/{}.png'.format(basename), label[0])
        cv2.waitKey(0)

    exit()

    # test get_batch
    for _ in range(100):
        batch_im, batch_label = ds.get_batch(10, use_enhance=True, window_hw=[168, 168])
        assert len(batch_im) == len(batch_label)
        for im, label in zip(batch_im, batch_label):
            cv2.imshow('ori_im', im)
            label = list(np.transpose(label, [2, 0, 1]).__mul__(255).astype(np.uint8))
            label = im_tool.draw_multi_img_in_big_img(label, 1, [400, 800], [2, 3], pad_color=64)
            cv2.imshow('label', label)
            cv2.waitKey(0)

    # test get_eval_batch_gen
    g = ds.get_eval_batch_gen(10, window_hw=[168, 168])
    for batch_im, batch_label in preload_generator.preload_generator(g):
        for im, label in zip(batch_im, batch_label):
            cv2.imshow('ori_im', im)
            label = list(np.transpose(label, [2, 1, 0]).__mul__(255).astype(np.uint8))
            label = im_tool.draw_multi_img_in_big_img(label, 1, [400, 800], [2, 3])
            cv2.imshow('label', label)
            cv2.waitKey(0)
