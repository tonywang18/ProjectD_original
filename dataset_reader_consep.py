import os
import imageio
import cv2
import numpy as np
import scipy.io
import random
import math
import gzip
import pickle
import copy
from a_config import project_root
from my_py_lib import contour_tool
from my_py_lib.image_over_scan_wrapper import ImageOverScanWrapper
from my_py_lib.coords_over_scan_gen import n_step_scan_coords_gen
from my_py_lib.list_tool import list_multi_get_with_ids
from my_py_lib.im_tool import check_and_tr_umat, ensure_image_has_3dim
import my_py_lib.color_enhance_tool as color_enhance_tool
from my_py_lib import point_tool
from my_py_lib.draw_repel_code_tool import draw_repel_code_fast
from skimage.transform import rotate as sk_rotate
from skimage.draw import disk as sk_disk


draw_repl_hm = draw_repel_code_fast


def draw_im_points(ori_im, ori_label: dict, n_cls, draw_rect, pt_radius_max, pt_radius_min, use_repel_code):
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
    # 使用 ImageOverScanWrapper 可以允许过采样
    wrap_im = ImageOverScanWrapper(ori_im)
    crop_im = wrap_im.get(draw_rect[:2], draw_rect[2:], 255)
    label = np.zeros([draw_rect[2] - draw_rect[0], draw_rect[3] - draw_rect[1], n_cls], np.float32)

    for k in ori_label.keys():
        pts = ori_label[k]
        if len(pts) == 0:
            continue
        pts = np.array(pts, np.int32)
        # 选出在绘画范围内的点
        is_in =  np.all([np.all(pts <= draw_rect[None, 2:], 1), np.all(pts >= draw_rect[None, :2], 1)], 0)
        in_pts = pts[is_in]
        if len(in_pts) == 0:
            continue

        # 平移选出的点
        in_pts = in_pts - draw_rect[None, :2]

        if not use_repel_code:
            # 原始画法
            for pt in in_pts:
                radius = random.uniform(pt_radius_min, pt_radius_max) if pt_radius_max > pt_radius_min else pt_radius_min
                radius = int(np.round(radius))
                rr, cc = sk_disk(pt, radius, shape=label.shape[:2])
                label[rr, cc, k] = 1
        else:
            # 新的画法，排斥编码
            draw_repl_hm(label[:, :, k:k+1], in_pts, pt_radius_max, A=0.5)

    return crop_im, label


class DatasetReader:
    # 原始类别
    ori_class_names = ('other', 'inflammatory', 'healthy epithelial', 'dysplastic/malignant epithelial', 'fibroblast', 'muscle', 'endothelial')
    # 原始类对应id
    ori_class_ids = (1, 2, 3, 4, 5, 6, 7)
    # 要合并的原始类别，键是待合并类，值是目标合并类
    ori_merge_class_dict = {7:5, 6:5}
    # 转换consep类别到pannuke类别
    tr_ori_class_id_dict = {1:3, 2:1, 3:4, 4:0, 5:2}
    # 临时用，融合 Epithelial类 到 Neoplastic类
    # tr_ori_class_id_dict = {1:3, 2:1, 3:0, 4:0, 5:2}

    class_names = ('Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial', 'Det')
    class_ids = (0, 1, 2, 3, 4, 5)
    class_num = len(class_ids)
    assert len(class_names) == len(class_ids)

    def __init__(self, dataset_path=os.path.join(project_root, 'consep_dataset'), data_type='train', pt_radius_min=3, pt_radius_max=3, use_repel_code=False, rescale=1., use_balance=True):
        data_type = '{}.pkl.gz'.format(data_type)
        self.data_path = os.path.join(dataset_path, data_type)
        self.pt_radius_min = pt_radius_min
        self.pt_radius_max = pt_radius_max
        self.use_repel_code = use_repel_code
        self.rescale = rescale
        self.use_balance = use_balance

        assert self.pt_radius_max >= self.pt_radius_min

        self._encode_im_list = []
        self._name_list = []
        self._label_list = []

        self._cla_label_id = dict([(i, []) for i in self.class_ids])

        self._build()

    def _build(self):
        all_data = pickle.loads(gzip.decompress(open(self.data_path, 'rb').read()))
        self._encode_im_list.extend(all_data['encode_ims'])
        self._name_list.extend(all_data['names'])
        ori_label_list = all_data['labels']

        # 转换类别
        new_label_list = []
        for ori_label in ori_label_list:
            new_label = {}

            for _c in self.class_ids:
                new_label[_c] = []

            for ori_cls in self.ori_class_ids:
                new_cls = ori_cls
                if new_cls in self.ori_merge_class_dict:
                    new_cls = self.ori_merge_class_dict[new_cls]
                new_cls = self.tr_ori_class_id_dict[new_cls]
                new_label[new_cls].extend(ori_label[ori_cls])

            new_label_list.append(new_label)

        self._label_list = new_label_list

        for i, label_cls_conts in enumerate(self._label_list):
            for c in self.class_ids:
                if len(label_cls_conts[c]) > 0:
                    self._cla_label_id[c].append(i)

        # 最终检查
        assert len(self._encode_im_list) == len(self._name_list) == len(self._label_list)

    def __len__(self):
        return len(self._encode_im_list)

    def get_item_point(self, item_id, use_enhance=False, window_hw=None, sample_center=None):
        window_hw = np.array(window_hw, np.int32) if window_hw is not None else window_hw
        encode_im = self._encode_im_list[item_id]
        name = self._name_list[item_id]
        label_cls_conts = self._label_list[item_id]

        im = cv2.imdecode(np.frombuffer(encode_im, np.uint8), -1)
        im = im[..., :3]
        im = im[..., ::-1]
        assert im.dtype == np.uint8

        if self.rescale != 1.:
            im = cv2.resize(im, (int(im.shape[0]*self.rescale), int(im.shape[1]*self.rescale)), interpolation=cv2.INTER_AREA)

        label_cls_center_pts = copy.deepcopy(label_cls_conts)
        for k in label_cls_center_pts.keys():
            for i in range(len(label_cls_center_pts[k])):
                cont = label_cls_center_pts[k][i]
                if self.rescale != 1.:
                    cont = contour_tool.resize_contours([cont], self.rescale)[0]
                box = contour_tool.make_bbox_from_contour(cont)
                center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.int32)
                label_cls_center_pts[k][i] = center

        more_info = {
            'im_id': item_id,
            'name': name,
            'label': label_cls_center_pts,
        }

        if window_hw is not None:
            if sample_center is not None:
                assert len(sample_center) == 2
                center_yx = np.asarray(sample_center, np.int32)
            else:
                # center_yx = np.random.randint(window_hw // 2, (im.shape[0], im.shape[1]) - window_hw // 2, 2, np.int32)
                center_yx = np.random.randint((0, 0), (im.shape[0], im.shape[1]), 2, np.int32)
            half_hw = window_hw // 2
            yx_start = center_yx - half_hw
            yx_end = center_yx + half_hw
            draw_rect = np.concatenate([yx_start, yx_end], 0)
        else:
            draw_rect = np.array([0, 0, im.shape[0], im.shape[1]])

        im, label_hm = draw_im_points(im, label_cls_center_pts, self.class_num, draw_rect, self.pt_radius_max, self.pt_radius_min, self.use_repel_code)

        if use_enhance:
            rot_angle = np.random.choice([0, 90, -90, 180])
            if rot_angle != -1:
                im = sk_rotate(im, rot_angle, preserve_range=True, order=0).astype(np.uint8)
                label_hm = sk_rotate(label_hm, rot_angle, preserve_range=True, order=0).astype(np.float32)
                hw = np.array(label_hm.shape[:2])
                for k in label_cls_center_pts.keys():
                    pts = np.asarray(label_cls_center_pts[k])
                    R = point_tool.make_rotate(rot_angle)
                    if len(pts) != 0:
                        pts = point_tool.apply_affine_to_points(pts, R)
                        # 去掉超出范围的点
                        keep_bools = np.logical_and(np.all(pts <= hw[None,], 1), np.all(pts >= np.array([0, 0])[None,], 1))
                        pts = pts[keep_bools]
                    label_cls_center_pts[k] = pts
                more_info['label'] = label_cls_center_pts
            # if np.random.uniform() > 0.7:
            #     # 随机高斯模糊
            #     ker_sz = 3 # np.random.choice([3, 5])
            #     o_im = cv2.GaussianBlur(o_im, (ker_sz, ker_sz), 0)
            if np.random.uniform() > 0.5:
                # 随机色调调整
                im = color_enhance_tool.random_adjust_HSV(im, 0.2, 0.2, 0.1)
            if np.random.uniform() > 0.5:
                # 随机噪音
                im = color_enhance_tool.random_gasuss_noise(im, 0, 0.05)

        im = ensure_image_has_3dim(im)
        label_hm = ensure_image_has_3dim(label_hm)

        return im, label_hm, more_info

    def get_train_batch_point(self, batch_size, use_enhance=False, window_hw=None):
        if self.use_balance:
            # 排除Det类
            ids = []
            assert batch_size >= self.class_num - 1
            each_cls_n = batch_size // (self.class_num - 1)
            for i in range(self.class_num-1):
                ids.extend(np.random.choice(self._cla_label_id[i], each_cls_n))
        else:
            ids = np.random.randint(0, len(self), batch_size)
        batch_im = []
        batch_label = []
        batch_info = []

        for i in ids:
            im, label, info = self.get_item_point(i, use_enhance, window_hw)
            batch_im.append(im)
            batch_label.append(label)
            batch_info.append(info)

        return np.asarray(batch_im), np.asarray(batch_label), batch_info

    def get_train_batch_point_gen(self, batch_count, batch_size, use_enhance=False, window_hw=None):
        for i in range(batch_count):
            batch_im, batch_label, batch_info = self.get_train_batch_point(batch_size, use_enhance=use_enhance, window_hw=window_hw)
            yield batch_im, batch_label, batch_info

    def get_eval_batch_point_gen(self, batch_size, window_hw):
        batch_im = []
        batch_label = []
        batch_info = []
        for i in range(len(self)):
            # 获取标签原图
            im, label, info = self.get_item_point(i, use_enhance=False)
            wim = ImageOverScanWrapper(im)
            wlabel = ImageOverScanWrapper(label)
            # 半步长采样
            coords_g = n_step_scan_coords_gen(im.shape[:2], window_hw, 1)
            for yx_start, yx_end in coords_g:
                patch_im = wim.get(yx_start, yx_end)
                patch_label = wlabel.get(yx_start, yx_end)
                batch_im.append(patch_im)
                batch_label.append(patch_label)

                if len(batch_im) >= batch_size:
                    yield np.asarray(batch_im), np.asarray(batch_label), batch_info
                    batch_im = []
                    batch_label = []
                    batch_info = []

        if len(batch_im) > 0:
            yield np.asarray(batch_im), np.asarray(batch_label), batch_info


if __name__ == '__main__':
    from my_py_lib import im_tool
    from my_py_lib import preload_generator

    ds = DatasetReader(data_type='train', pt_radius_min=9, pt_radius_max=9, use_repel_code=True, rescale=0.5)

    # # test get_item
    # for i in range(len(ds)):
    #     im, label, info = ds.get_item_point(i, use_enhance=False, window_hw=[64, 64])
    #     cv2.imshow('ori_im', cv2.resize(im[..., ::-1], (128, 128), interpolation=cv2.INTER_AREA))
    #     label1 = np.transpose(label, [2, 0, 1]).__mul__(255).astype(np.uint8)
    #     label1 = im_tool.draw_multi_img_in_big_img(label1, 1, [400, 800], [2, 3], pad_color=64)
    #     cv2.imshow('label1', label1)
    #
    #     draw_im = im
    #     for i in range(label.shape[2]):
    #         cons = contour_tool.find_contours((label[..., i] > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL,
    #                                           cv2.CHAIN_APPROX_SIMPLE)
    #         draw_im = contour_tool.draw_contours(draw_im, cons, (255, 0, 0))
    #     cv2.imshow('draw_im', draw_im[..., ::-1])
    #
    #     # os.makedirs('tmp', exist_ok=True)
    #     # basename = info['im_ids']
    #     # imageio.imwrite('tmp/{}.png'.format(basename), label[0])
    #     cv2.waitKey(0)

    # exit()

    # test get_batch
    for _ in range(100):
        batch_im, batch_label, batch_info = ds.get_train_batch_point(10, use_enhance=True, window_hw=[168, 168])
        assert len(batch_im) == len(batch_label)
        for im, label in zip(batch_im, batch_label):
            cv2.imshow('ori_im', im[..., ::-1])

            draw_im = im
            for i in range(label.shape[2]):
                cons = contour_tool.find_contours((label[..., i] > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                draw_im = contour_tool.draw_contours(draw_im, cons, (255, 0, 0))
            cv2.imshow('draw_im', draw_im[..., ::-1])

            label = list(np.transpose(label, [2, 0, 1]).__mul__(255).astype(np.uint8))
            label = im_tool.draw_multi_img_in_big_img(label, 1, [400, 800], [2, 3], pad_color=64)
            cv2.imshow('label', label)

            cv2.waitKey(0)

    # test get_eval_batch_gen
    g = ds.get_eval_batch_point_gen(10, window_hw=[168, 168])
    for batch_im, batch_label, batch_info in preload_generator.preload_generator(g):
        for im, label in zip(batch_im, batch_label):
            cv2.imshow('ori_im', im[..., ::-1])
            label = list(np.transpose(label, [2, 1, 0]).__mul__(255).astype(np.uint8))
            label = im_tool.draw_multi_img_in_big_img(label, 1, [400, 800], [2, 3])
            cv2.imshow('label', label)
            cv2.waitKey(0)
