import os
import torch
import torch.nn.functional as F
import time
import numpy as np
import yaml
from dataset_reader2 import DatasetReader
from tensorboardX import SummaryWriter
import cv2
import imageio
import copy
from my_py_lib import im_tool
from my_py_lib import contour_tool
from my_py_lib.auto_show_running import AutoShowRunning

from a_config import project_root, device, net_in_hw, net_out_hw, batch_size, epoch, batch_count, eval_which_checkpoint,\
    dataset_path, process_control, net_save_dir, match_distance_thresh_list, make_cla_is_det
from eval_utils import calc_a_sample_info_points_each_class, class_map_to_contours2
from my_py_lib.preload_generator import preload_generator
from my_py_lib.numpy_tool import one_hot
from my_py_lib.image_over_scan_wrapper import ImageOverScanWrapper
from my_py_lib.coords_over_scan_gen import n_step_scan_coords_gen
from my_py_lib import list_tool
from eval_utils import calc_a_sample_info
import eval_utils
import visdom
from big_pic_result import BigPicPatch
from heatmap_nms import heatmap_nms
from tsp_cnn.net import TSPCNN


use_heatmap_nms = False
use_single_pair = True


in_dim = 3
# device = torch.device(0)
# net_in_hw = (320, 320)
# net_out_hw = (160, 160)
# batch_size = 3
# epoch = 1000
# batch_count = 5
auto_show_interval = 4

if make_cla_is_det:
    cla_class_num = 1
    cla_class_ids = list(range(cla_class_num))
    cla_class_names = DatasetReader.class_names[0:1]
else:
    cla_class_num = DatasetReader.class_num - 1
    cla_class_ids = list(range(cla_class_num))
    cla_class_names = DatasetReader.class_names[1:]


pred_train_out_dir = 'pred_circle_train_out_dir_tspcnn'
pred_valid_out_dir = 'pred_circle_valid_out_dir_tspcnn'
pred_test_out_dir = 'pred_circle_test_out_dir_tspcnn'


'''
命名注意
eval是评估
valid是验证集
test是测试集
eval可以是验证集，也可以是测试集

cm 代表类别图
pm 代表概率图
'''

nrow = 12


def show_ims(vtb: visdom.Visdom, tensors, name):
    assert tensors.ndim == 4
    for c in range(tensors.shape[1]):
        n1 = f'{name}_{c}'
        vtb.images(tensors[:, c:c+1], nrow=nrow, win=n1, opts={'title': n1})


def tr_pm_to_onehot_vec(label_pm: torch.Tensor):
    label_oh = label_pm
    bg = 1 - label_oh.max(1, keepdim=True)[0]
    label_oh = torch.cat([bg, label_oh], dim=1)
    return label_oh


# def tr_pm_to_onehot_vec(label_pm: torch.Tensor):
#     label_oh = (label_pm > 0.5).type(torch.float32)
#     bg = 1 - label_oh.max(1, keepdim=True)[0]
#     label_oh = torch.cat([bg, label_oh], dim=1)
#     return label_oh


def tr_cla_vec_to_det_vec(label_cla_ce: torch.Tensor):
    bg = (label_cla_ce[:, 0:1] > 0.5).type(torch.float32)
    pos = 1 - bg
    det_vec = torch.cat([bg, pos], dim=1)
    return det_vec


def get_pg_id(epoch, process_control):
    if epoch < process_control[0]:
        pg_id = 1
    elif process_control[0] <= epoch < process_control[1]:
        pg_id = 2
    elif process_control[1] <= epoch:
        pg_id = 3
    else:
        raise AssertionError()
    return pg_id


def get_pg_name(ori_name, start_epoch):
    pg_id = get_pg_id(start_epoch, process_control)
    base, ext = os.path.splitext(ori_name)
    out_name = base + '_p{}'.format(pg_id) + ext
    return out_name


def make_mix_pic(im, pred, label, prob=0.4):
    color_label = (label > 0.5).astype(np.uint8) * 255
    color_pred = (pred > prob).astype(np.uint8) * 255
    pad = np.zeros([*color_pred.shape[:2], 1], dtype=color_pred.dtype)
    color_hm = np.concatenate([color_pred, color_label, pad], axis=-1)
    mix_pic = np.where(np.any(color_hm > 0, -1, keepdims=True), color_hm, im)
    return mix_pic


def main(net):

    torch.set_grad_enabled(False)
    _last_auto_show_update_time = 0

    train_dataset = DatasetReader(dataset_path, 'train', pt_radius_min=3, pt_radius_max=3)
    valid_dataset = DatasetReader(dataset_path, 'valid', pt_radius_min=3, pt_radius_max=3)
    test_dataset = DatasetReader(dataset_path, 'test', pt_radius_min=3, pt_radius_max=3)

    net = net.to(device)
    net.eval()

    # 分割相关
    for did, cur_dataset in enumerate([train_dataset, valid_dataset, test_dataset]):
        out_dir = [pred_train_out_dir, pred_valid_out_dir, pred_test_out_dir][did]

        os.makedirs(out_dir, exist_ok=True)

        det_score_table = {}
        for dt in match_distance_thresh_list:
            det_score_table[dt] = {
                'found_pred': 0,  # 所有的假阳性
                'fakefound_pred': 0,  # 所有的假阴性
                'found_label': 0,  # 所有找到的标签
                'nofound_label': 0,  # 所有找到的预测
                'label_repeat': 0,  # 对应了多个pred的标签
                'pred_repeat': 0,  # 对应了多个label的预测
                'f1': None,
                'recall': None,
                'prec': None,
            }

        for pid in range(len(cur_dataset)):
            im, label, ignore_mask, info = cur_dataset.get_item(pid, use_enhance=False, window_hw=None)

            label_det, label_cla = np.split(label, [1, ], -1)

            label_det_pts = info['label_det_pts']
            label_det_pts_without_ignore = info['label_det_pts_without_ignore']
            group_label_cla_pts = info['group_label_cla_pts']
            group_label_cla_pts_without_ignore = info['group_label_cla_pts_without_ignore']
            ignore_pts = info['ignore_pts']

            if make_cla_is_det:
                label_det_pts_without_ignore = label_det_pts
                group_label_cla_pts = {0: label_det_pts}
                group_label_cla_pts_without_ignore = group_label_cla_pts
                ignore_pts = []

            im_path = info['im_path']

            im_basename = os.path.splitext(os.path.basename(im_path))[0]

            print('Processing {}'.format(im_basename))

            del label

            # 运行区
            bpp_im = BigPicPatch(1, [im], (0, 0), window_hw=net_in_hw, level_0_patch_hw=(1, 1), custom_patch_merge_pipe=eval_utils.patch_merge_func, patch_border_pad_value=255)
            for batch_info, batch_patch in bpp_im.batch_get_im_patch_gen(batch_size):

                tmp = []
                for tmp_im in batch_patch:
                    tmp.append(cv2.cvtColor(tmp_im, cv2.COLOR_RGB2GRAY))
                tmp = np.asarray(tmp)[..., None]
                batch_patch = tmp

                batch_patch = torch.tensor(np.array(batch_patch), dtype=torch.float32, device=device) / 255
                batch_patch = batch_patch.permute(0, 3, 1, 2)
                batch_pred_det = net(batch_patch)

                out_patch_pred_det = batch_pred_det.clamp(0, 1)

                out_patch_pred_det = out_patch_pred_det.permute(0, 2, 3, 1).cpu().numpy()

                out_pred = out_patch_pred_det

                bpp_im.batch_update_result(batch_info, out_pred)

            pred = bpp_im.multi_scale_result[0].data / bpp_im.multi_scale_mask[0].data
            pred_det, pred_cla = np.split(pred, [1], 2)

            # det
            if use_heatmap_nms:
                for c in range(pred_det.shape[2]):
                    pred_det[:, :, c] = heatmap_nms(pred_det[:, :, c])
            pred_det_pts = eval_utils.get_pts_from_hm(pred_det, 0.3)
            # label_det_pts = eval_utils.get_pts_from_hm(label_det, 0.5)

            mix_pic_det_a1 = eval_utils.draw_hm_circle(im, pred_det_pts, label_det_pts, 6)

            det_info = calc_a_sample_info_points_each_class([pred_det_pts, [0]*len(pred_det_pts)], [label_det_pts, [0]*len(label_det_pts)], [0], match_distance_thresh_list, use_single_pair=use_single_pair)

            for dt in match_distance_thresh_list:
                for cls in [0]:
                    det_score_table[dt]['found_pred'] += det_info[cls][dt]['found_pred']
                    det_score_table[dt]['fakefound_pred'] += det_info[cls][dt]['fakefound_pred']
                    det_score_table[dt]['found_label'] += det_info[cls][dt]['found_label']
                    det_score_table[dt]['nofound_label'] += det_info[cls][dt]['nofound_label']
                    det_score_table[dt]['pred_repeat'] += det_info[cls][dt]['pred_repeat']
                    det_score_table[dt]['label_repeat'] += det_info[cls][dt]['label_repeat']

            imageio.imwrite(os.path.join(out_dir, '{}_1det_a1_m.png'.format(im_basename)), mix_pic_det_a1)
            imageio.imwrite(os.path.join(out_dir, '{}_1det_a1_h.png'.format(im_basename)), (pred_det * 255).astype(np.uint8))
            yaml.dump(det_info, open(os.path.join(out_dir, '{}_det.txt'.format(im_basename)), 'w'))

        # 计算det a1 F1，精确率，召回率
        for dt in match_distance_thresh_list:
            prec = det_score_table[dt]['found_pred'] / (det_score_table[dt]['found_pred'] + det_score_table[dt]['fakefound_pred'] + 1e-8)
            recall = det_score_table[dt]['found_label'] / (det_score_table[dt]['found_label'] + det_score_table[dt]['nofound_label'] + 1e-8)
            f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
            det_score_table[dt]['prec'] = prec
            det_score_table[dt]['recall'] = recall
            det_score_table[dt]['f1'] = f1

        yaml.dump(det_score_table, open(os.path.join(out_dir, 'all_det.txt'), 'w'))


if __name__ == '__main__':
    # from main_net7 import MainNet
    net = TSPCNN()
    net.load_state_dict(torch.load('tsp_cnn/tsp_cnn.pt', 'cpu'))
    main(net)

