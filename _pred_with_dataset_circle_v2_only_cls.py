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
    dataset_path, process_control, net_save_dir, match_distance_thresh_list, b1_is_ce_loss, b2_is_ce_loss, make_cla_is_det, save_postfix
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


get_pg_id = eval_utils.get_pg_id
get_pg_name = eval_utils.get_pg_name

use_heatmap_nms = False
use_single_pair = True
make_cla_is_det = False


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


pred_train_out_dir = 'only_cls_pred_circle_train_out_dir{}'.format(save_postfix)
pred_valid_out_dir = 'only_cls_pred_circle_valid_out_dir{}'.format(save_postfix)
pred_test_out_dir = 'only_cls_pred_circle_test_out_dir{}'.format(save_postfix)


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


def make_mix_pic(im, pred, label, prob=0.4):
    color_label = (label > 0.5).astype(np.uint8) * 255
    color_pred = (pred > prob).astype(np.uint8) * 255
    pad = np.zeros([*color_pred.shape[:2], 1], dtype=color_pred.dtype)
    color_hm = np.concatenate([color_pred, color_label, pad], axis=-1)
    mix_pic = np.where(np.any(color_hm > 0, -1, keepdims=True), color_hm, im)
    return mix_pic



def main(NetClass):

    torch.set_grad_enabled(False)
    _last_auto_show_update_time = 0

    model_id = NetClass.model_id

    ck_name         = '{}/{}_model.pt'          .format(net_save_dir, model_id)
    ck_best_name    = '{}/{}_model_best.pt'     .format(net_save_dir, model_id)
    ck_minloss_name = '{}/{}_model_minloss.pt'     .format(net_save_dir, model_id)
    # ck_optim_name   = '{}/{}_optim.pt'          .format(net_save_dir, model_id)
    # ck_restart_name = '{}/{}_restart.yml'       .format(net_save_dir, model_id)
    # ck_extra_name   = '{}/{}_extra.txt'         .format(net_save_dir, model_id)
    # score_name      = '{}/{}_score.txt'         .format(net_save_dir, model_id)
    # score_best_name = '{}/{}_score_best.txt'    .format(net_save_dir, model_id)

    pt_radius = 3
    train_dataset = DatasetReader(dataset_path, 'train', pt_radius_min=pt_radius, pt_radius_max=pt_radius)
    valid_dataset = DatasetReader(dataset_path, 'valid', pt_radius_min=pt_radius, pt_radius_max=pt_radius)
    test_dataset = DatasetReader(dataset_path, 'test', pt_radius_min=pt_radius, pt_radius_max=pt_radius)

    # 定义网络
    net = NetClass(3, cla_class_num)

    if eval_which_checkpoint == 'last':
        print('Will load last weight.')
        net.load_state_dict(torch.load(ck_name, 'cpu'))
        print('load model success')
    elif eval_which_checkpoint == 'best':
        print('Will load best weight.')
        net.load_state_dict(torch.load(ck_best_name, 'cpu'))
        print('load model success')
    elif eval_which_checkpoint == 'minloss':
        print('Will load minloss weight.')
        net.load_state_dict(torch.load(ck_minloss_name, 'cpu'))
        print('load model success')
    else:
        print('Unknow weight type. Will not load weight.')

    net = net.to(device)
    net.eval()

    # 分割相关
    for did, cur_dataset in enumerate([train_dataset, valid_dataset, test_dataset]):
        out_dir = [pred_train_out_dir, pred_valid_out_dir, pred_test_out_dir][did]

        # if did != 1:
        #     continue

        os.makedirs(out_dir, exist_ok=True)

        cla_score_table = {}
        for dt in match_distance_thresh_list:
            cla_score_table[dt] = {
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

            for cls_id in cla_class_ids:
                cla_score_table[dt][cls_id] = {
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

            im_path = info['im_path']

            im_basename = os.path.splitext(os.path.basename(im_path))[0]

            print('Processing {}'.format(im_basename))

            # label = np.asarray(label, np.float32)
            # label_det, label_cla = np.split(label, [1,], 2)
            # label_mask = make_unmatched_mask_np(label_det, label_cla)
            # del label

            # # test B
            # tmp_show = './tmp_show'
            # os.makedirs(tmp_show, exist_ok=True)

            # pack_im_label = np.concatenate([im, label], 2)

            # 运行区
            bpp_im = BigPicPatch(cla_class_num, [im], (0, 0), net_in_hw, (1, 1), 0, 0.4, custom_patch_merge_pipe=eval_utils.patch_merge_func, patch_border_pad_value=255)
            for batch_info, batch_patch in bpp_im.batch_get_im_patch_gen(batch_size):

                batch_patch_im = torch.tensor(np.array(batch_patch), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255
                batch_patch_pred_cla = net(batch_patch_im)

                # if b2_is_ce_loss:
                #     batch_patch_pred_cla = batch_patch_pred_cla.softmax(1)[:, 1:]
                # else:
                batch_patch_pred_cla = batch_patch_pred_cla.clamp(0, 1)

                batch_pred = batch_patch_pred_cla.permute(0, 2, 3, 1).cpu().numpy()

                bpp_im.batch_update_result(batch_info, batch_pred)

                # # test B
                # for i, info in enumerate(batch_info):
                #     level, yx_start, yx_end = info
                #     out_im = os.path.join(tmp_show, '{}_{}_m.png'.format(yx_start[0], yx_start[1]))
                #     imageio.imwrite(out_im, tmp1[i])
                #
                #     for c in range(out_pred.shape[3]):
                #         if c != 0:
                #             continue
                #         out_im = os.path.join(tmp_show, '{}_{}_{}.png'.format(yx_start[0], yx_start[1], c))
                #         imageio.imwrite(out_im, out_pred[i, ..., c])

            pred = bpp_im.multi_scale_result[0].data / np.clip(bpp_im.multi_scale_mask[0].data, 1e-8, None)
            pred_cla = pred
            pred_cla_final = pred_cla * np.max(label_cla, axis=2, keepdims=True)

            # cla
            if use_heatmap_nms:
                for c in range(pred_cla_final.shape[2]):
                    pred_cla_final[:, :, c] = pred_cla_final[:, :, c] * heatmap_nms(pred_cla_final[:, :, c])

            pred_cla_pts = []
            pred_cla_pts_cla = []
            for i in range(pred_cla_final.shape[2]):
                pts = eval_utils.get_pts_from_hm(pred_cla_final[..., i:i+1], 0.1)
                pts = list_tool.list_multi_get_with_ids(pts, eval_utils.find_too_far_pts(pts, ignore_pts))
                pred_cla_pts.extend(pts)
                pred_cla_pts_cla.extend([i]*len(pts))

            label_cla_pts = []
            label_cla_pts_cla = []
            for i in group_label_cla_pts_without_ignore:
                label_cla_pts.extend(group_label_cla_pts_without_ignore[i])
                if make_cla_is_det:
                    label_cla_pts_cla.extend([i]*len(group_label_cla_pts_without_ignore[i]))
                else:
                    label_cla_pts_cla.extend([i-1]*len(group_label_cla_pts_without_ignore[i]))

            cla_info = calc_a_sample_info_points_each_class([pred_cla_pts, pred_cla_pts_cla], [label_cla_pts, label_cla_pts_cla], cla_class_ids,
                                                            match_distance_thresh_list, use_single_pair=use_single_pair)

            for dt in match_distance_thresh_list:
                for cls in cla_class_ids:
                    cla_score_table[dt][cls]['found_pred'] += cla_info[cls][dt]['found_pred']
                    cla_score_table[dt][cls]['fakefound_pred'] += cla_info[cls][dt]['fakefound_pred']
                    cla_score_table[dt][cls]['found_label'] += cla_info[cls][dt]['found_label']
                    cla_score_table[dt][cls]['nofound_label'] += cla_info[cls][dt]['nofound_label']
                    cla_score_table[dt][cls]['pred_repeat'] += cla_info[cls][dt]['pred_repeat']
                    cla_score_table[dt][cls]['label_repeat'] += cla_info[cls][dt]['label_repeat']

                    cla_score_table[dt]['found_pred'] += cla_info[cls][dt]['found_pred']
                    cla_score_table[dt]['fakefound_pred'] += cla_info[cls][dt]['fakefound_pred']
                    cla_score_table[dt]['found_label'] += cla_info[cls][dt]['found_label']
                    cla_score_table[dt]['nofound_label'] += cla_info[cls][dt]['nofound_label']
                    cla_score_table[dt]['pred_repeat'] += cla_info[cls][dt]['pred_repeat']
                    cla_score_table[dt]['label_repeat'] += cla_info[cls][dt]['label_repeat']

            group_mix_pic_cla = []

            for i in range(cla_class_num):
                cur_pred_pts = list_tool.list_multi_get_with_bool(pred_cla_pts, np.array(pred_cla_pts_cla, np.int) == i)
                cur_label_pts = list_tool.list_multi_get_with_bool(label_cla_pts, np.array(label_cla_pts_cla, np.int) == i)
                group_mix_pic_cla.append(eval_utils.draw_hm_circle(im, cur_pred_pts, cur_label_pts, 6))

            # 输出分类分支的图
            for i in range(cla_class_num):
                imageio.imwrite(os.path.join(out_dir, '{}_2cla_b1_{}.png'.format(im_basename, cla_class_names[i])), group_mix_pic_cla[i])

                # 生成分类分支的处理前和处理后的热图
                imageio.imwrite(os.path.join(out_dir, '{}_2cla_b2_{}_1before.png'.format(im_basename, cla_class_names[i])), (pred_cla[..., i] * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(out_dir, '{}_2cla_b2_{}_2after.png'.format(im_basename, cla_class_names[i])), (pred_cla_final[..., i] * 255).astype(np.uint8))

        # 计算cla F1，精确率，召回率
        for dt in match_distance_thresh_list:
            prec = cla_score_table[dt]['found_pred'] / (cla_score_table[dt]['found_pred'] + cla_score_table[dt]['fakefound_pred'] + 1e-8)
            recall = cla_score_table[dt]['found_label'] / (cla_score_table[dt]['found_label'] + cla_score_table[dt]['nofound_label'] + 1e-8)
            f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
            cla_score_table[dt]['prec'] = prec
            cla_score_table[dt]['recall'] = recall
            cla_score_table[dt]['f1'] = f1

            for cls_id in cla_class_ids:
                prec = cla_score_table[dt][cls_id]['found_pred'] / (cla_score_table[dt][cls_id]['found_pred'] + cla_score_table[dt][cls_id]['fakefound_pred'] + 1e-8)
                recall = cla_score_table[dt][cls_id]['found_label'] / (cla_score_table[dt][cls_id]['found_label'] + cla_score_table[dt][cls_id]['nofound_label'] + 1e-8)
                f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
                cla_score_table[dt][cls_id]['prec'] = prec
                cla_score_table[dt][cls_id]['recall'] = recall
                cla_score_table[dt][cls_id]['f1'] = f1

        yaml.dump(cla_score_table, open(os.path.join(out_dir, 'all_cla.txt'), 'w'))


if __name__ == '__main__':
    from main_net9 import MainNet2 as MainNet
    main(MainNet)

