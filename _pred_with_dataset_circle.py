import os
import torch
import torch.nn.functional as F
import time
import numpy as np
import yaml
from dataset_reader import DatasetReader
from tensorboardX import SummaryWriter
import cv2
import imageio
import copy
from my_py_lib import im_tool
from my_py_lib import contour_tool
from my_py_lib.auto_show_running import AutoShowRunning

from a_config import project_root, device, net_in_hw, net_out_hw, batch_size, epoch, batch_count, eval_which_checkpoint, dataset_path, process_control, net_save_dir, match_distance_thresh_list
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


in_dim = 3
# device = torch.device(0)
# net_in_hw = (320, 320)
# net_out_hw = (160, 160)
# batch_size = 3
# epoch = 1000
# batch_count = 5
auto_show_interval = 4

cla_class_num = DatasetReader.cla_class_num
cla_class_ids = DatasetReader.cla_class_ids
cla_class_names = DatasetReader.cla_class_names


pred_det_train_out_dir = 'pred_circle_det_train_out_dir'
pred_det_valid_out_dir = 'pred_circle_det_valid_out_dir'
pred_cla_train_out_dir = 'pred_circle_cla_train_out_dir'
pred_cla_valid_out_dir = 'pred_circle_cla_valid_out_dir'


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
    label_oh = (label_pm > 0.5).type(torch.float32)
    bg = 1 - label_oh.max(1, keepdim=True)[0]
    label_oh = torch.cat([bg, label_oh], dim=1)
    return label_oh


def tr_cla_vec_to_det_vec(label_cla_ce: torch.Tensor):
    bg = (label_cla_ce[:, 0:1] > 0.5).type(torch.float32)
    pos = 1 - bg
    det_vec = torch.cat([bg, pos], dim=1)
    return det_vec


def nms_pm(pm: np.ndarray):
    bm = (pm > pm.max(2, keepdims=True)).astype(pm.dtype)
    pm = pm * bm
    return pm


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

    start_epoch = 123

    pg_id = get_pg_id(start_epoch, process_control)

    # os.makedirs(pred_det_train_out_dir, exist_ok=True)
    # os.makedirs(pred_det_valid_out_dir, exist_ok=True)
    # os.makedirs(pred_cla_train_out_dir, exist_ok=True)
    # os.makedirs(pred_cla_valid_out_dir, exist_ok=True)

    train_det_dataset = DatasetReader(dataset_path, 'det', is_train=True, pt_radius_min=3, pt_radius_max=3)
    eval_det_dataset = DatasetReader(dataset_path, 'det', is_train=False, pt_radius_min=3, pt_radius_max=3)

    train_cla_dataset = DatasetReader(dataset_path, 'cla', is_train=True, pt_radius_min=3, pt_radius_max=3)
    eval_cla_dataset = DatasetReader(dataset_path, 'cla', is_train=False, pt_radius_min=3, pt_radius_max=3)

    net = NetClass()

    # if pg_id == 1:
    #     net.enabled_cls_branch = False
    # else:
    #     net.enabled_cls_branch = True
    net.enabled_cls_branch = True

    if eval_which_checkpoint == 'last':
        print('Will load last weight.')
        new_ck_name = get_pg_name(ck_name, start_epoch)
        net.load_state_dict(torch.load(new_ck_name, 'cpu'))
        print('load model success')
    elif eval_which_checkpoint == 'best':
        print('Will load best weight.')
        new_ck_name = get_pg_name(ck_best_name, start_epoch)
        net.load_state_dict(torch.load(new_ck_name, 'cpu'))
        print('load model success')
    elif eval_which_checkpoint == 'minloss':
        print('Will load minloss weight.')
        new_ck_name = get_pg_name(ck_minloss_name, start_epoch)
        net.load_state_dict(torch.load(new_ck_name, 'cpu'))
        print('load model success')
    else:
        print('Unknow weight type. Will not load weight.')

    net = net.to(device)
    net.eval()

    # 分割相关
    for did, cur_dataset in enumerate([train_det_dataset, eval_det_dataset, train_cla_dataset, eval_cla_dataset]):
        out_dir = [pred_det_train_out_dir, pred_det_valid_out_dir, pred_cla_train_out_dir, pred_cla_valid_out_dir][did]

        # # 跳过det数据集
        # if did < 2:
        #     continue

        os.makedirs(out_dir, exist_ok=True)

        if pg_id == 1 and did >= 2:
            break

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
            
        det_score_table_a2 = copy.deepcopy(det_score_table)

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

            for cls_id in DatasetReader.cla_class_ids:
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
            im, label, im_info = cur_dataset.get_item(pid, use_enhance=False, window_hw=None, return_im_info=True)

            im_path = im_info['im_path']
            im_basename = os.path.splitext(os.path.basename(im_path))[0]

            label = np.asarray(label, np.float32)
            if did < 2:
                label_det = label
                label_cla = None
            else:
                label_det = label.max(axis=-1, keepdims=True)
                label_cla = label
            del label

            # 运行区
            bpp_im = BigPicPatch(1+5, [im], (0, 0), window_hw=net_in_hw, level_0_patch_hw=(1, 1), custom_patch_merge_pipe=eval_utils.patch_merge_func)
            for batch_info, batch_patch in bpp_im.batch_get_im_patch_gen(batch_size):

                batch_patch = torch.tensor(np.array(batch_patch), dtype=torch.float32, device=device) / 255
                batch_patch = batch_patch.permute(0, 3, 1, 2)
                batch_pred_det, batch_pred_cla = net(batch_patch)

                out_patch_pred_det = batch_pred_det.softmax(1)[:, 1:2]
                out_patch_pred_det = out_patch_pred_det.permute(0, 2, 3, 1).cpu().numpy()

                # out_patch_pred_a2 = batch_pred_det.softmax(1)[:, 1:2] * (1 - batch_pred_cla.softmax(1)[:, 0:1])
                # out_patch_pred_a2 = out_patch_pred_a2.permute(0, 2, 3, 1).cpu().numpy()

                # out_patch_pred_cla = batch_pred_det.softmax(1)[:, 1:2] * batch_pred_cla.softmax(1)[:, 1:]
                # out_patch_pred_cla = out_patch_pred_cla.permute(0, 2, 3, 1).cpu().numpy()

                out_patch_pred_cla = batch_pred_cla.softmax(1)
                out_patch_pred_cla = out_patch_pred_cla.permute(0, 2, 3, 1).cpu().numpy()

                out_pred = np.concatenate([out_patch_pred_det, out_patch_pred_cla], 3)

                bpp_im.batch_update_result(batch_info, out_pred)

            pred = bpp_im.multi_scale_result[0].data / bpp_im.multi_scale_mask[0].data
            pred_det, pred_cla = np.split(pred, [1], 2)

            # det
            pred_det_pts = eval_utils.get_pts_from_hm(pred_det, 0.5)
            label_det_pts = eval_utils.get_pts_from_hm(label_det, 0.5)

            mix_pic_det_a1 = eval_utils.draw_hm_circle(im, pred_det_pts, label_det_pts, 6)

            det_info = calc_a_sample_info_points_each_class([pred_det_pts, [0]*len(pred_det_pts)], [label_det_pts, [0]*len(label_det_pts)], [0], match_distance_thresh_list)

            for dt in match_distance_thresh_list:
                for cls in [0]:
                    det_score_table[dt]['found_pred'] += det_info[cls][dt]['found_pred']
                    det_score_table[dt]['fakefound_pred'] += det_info[cls][dt]['fakefound_pred']
                    det_score_table[dt]['found_label'] += det_info[cls][dt]['found_label']
                    det_score_table[dt]['nofound_label'] += det_info[cls][dt]['nofound_label']
                    det_score_table[dt]['pred_repeat'] += det_info[cls][dt]['pred_repeat']
                    det_score_table[dt]['label_repeat'] += det_info[cls][dt]['label_repeat']

            imageio.imwrite(os.path.join(out_dir, '{}_det_a1.png'.format(im_basename)), mix_pic_det_a1)
            imageio.imwrite(os.path.join(out_dir, '{}_det_a1_h.png'.format(im_basename)), (pred_det * 255).astype(np.uint8))
            yaml.dump(det_info, open(os.path.join(out_dir, '{}_det.txt'.format(im_basename)), 'w'))

            # cla
            # pred_det_a2 = pred_det * (1-pred_cla[..., :1])

            if did >= 2:

                pred_det_post = pred_det * (1 - pred_cla[..., :1])
                pred_det_post_pts = eval_utils.get_pts_from_hm(pred_det_post, 0.5)

                mix_pic_det_a2 = eval_utils.draw_hm_circle(im, pred_det_post_pts, label_det_pts, 6)

                det_info_a2 = calc_a_sample_info_points_each_class([pred_det_post_pts, [0] * len(pred_det_post_pts)],
                                                                    [label_det_pts, [0] * len(label_det_pts)], [0],
                                                                    match_distance_thresh_list)

                for dt in match_distance_thresh_list:
                    for cls in [0]:
                        det_score_table_a2[dt]['found_pred'] += det_info_a2[cls][dt]['found_pred']
                        det_score_table_a2[dt]['fakefound_pred'] += det_info_a2[cls][dt]['fakefound_pred']
                        det_score_table_a2[dt]['found_label'] += det_info_a2[cls][dt]['found_label']
                        det_score_table_a2[dt]['nofound_label'] += det_info_a2[cls][dt]['nofound_label']
                        det_score_table_a2[dt]['pred_repeat'] += det_info_a2[cls][dt]['pred_repeat']
                        det_score_table_a2[dt]['label_repeat'] += det_info_a2[cls][dt]['label_repeat']

                pred_cla_post = pred_det * pred_cla[..., 1:]
                pred_cla_pts = []
                pred_cla_pts_cla = []
                for i in range(pred_cla_post.shape[2]):
                    pts = eval_utils.get_pts_from_hm(pred_cla_post[..., i:i+1], 0.5)
                    pred_cla_pts.extend(pts)
                    pred_cla_pts_cla.extend([i]*len(pts))

                label_cla_pts = []
                label_cla_pts_cla = []
                for i in range(label_cla.shape[2]):
                    pts = eval_utils.get_pts_from_hm(label_cla[..., i:i+1], 0.5)
                    label_cla_pts.extend(pts)
                    label_cla_pts_cla.extend([i]*len(pts))

                cla_info = calc_a_sample_info_points_each_class([pred_cla_pts, pred_cla_pts_cla], [label_cla_pts, label_cla_pts_cla], [0, 1, 2, 3], match_distance_thresh_list)

                for dt in match_distance_thresh_list:
                    for cls in DatasetReader.cla_class_ids:
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

                for i in range(label_cla.shape[2]):
                    cur_pred_pts = list_tool.list_multi_get_with_bool(pred_cla_pts, np.array(pred_cla_pts_cla, np.int) == i)
                    cur_label_pts = list_tool.list_multi_get_with_bool(label_cla_pts, np.array(label_cla_pts_cla, np.int) == i)
                    group_mix_pic_cla.append(eval_utils.draw_hm_circle(im, cur_pred_pts, cur_label_pts, 6))

                imageio.imwrite(os.path.join(out_dir, '{}_det_a2.png'.format(im_basename)), mix_pic_det_a2)
                imageio.imwrite(os.path.join(out_dir, '{}_det_a2_h.png'.format(im_basename)), (pred_det_post * 255).astype(np.uint8))
                yaml.dump(det_info, open(os.path.join(out_dir, '{}_det_a2.txt'.format(im_basename)), 'w'))

                for i in range(4):
                    imageio.imwrite(os.path.join(out_dir, '{}_cla_b1_{}.png'.format(im_basename, DatasetReader.cla_class_names[i])), group_mix_pic_cla[i])

        # 计算det F1，精确率，召回率
        for dt in match_distance_thresh_list:
            prec = det_score_table[dt]['found_pred'] / (det_score_table[dt]['found_pred'] + det_score_table[dt]['fakefound_pred'] + 1e-8)
            recall = det_score_table[dt]['found_label'] / (det_score_table[dt]['found_label'] + det_score_table[dt]['nofound_label'] + 1e-8)
            f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
            det_score_table[dt]['prec'] = prec
            det_score_table[dt]['recall'] = recall
            det_score_table[dt]['f1'] = f1

        yaml.dump(det_score_table, open(os.path.join(out_dir, 'all_det.txt'), 'w'))

        # 计算det a2 F1，精确率，召回率
        for dt in match_distance_thresh_list:
            prec = det_score_table_a2[dt]['found_pred'] / (det_score_table_a2[dt]['found_pred'] + det_score_table_a2[dt]['fakefound_pred'] + 1e-8)
            recall = det_score_table_a2[dt]['found_label'] / (det_score_table_a2[dt]['found_label'] + det_score_table_a2[dt]['nofound_label'] + 1e-8)
            f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
            det_score_table_a2[dt]['prec'] = prec
            det_score_table_a2[dt]['recall'] = recall
            det_score_table_a2[dt]['f1'] = f1

        yaml.dump(det_score_table_a2, open(os.path.join(out_dir, 'all_det_a2.txt'), 'w'))

        # 计算cla F1，精确率，召回率
        for dt in match_distance_thresh_list:
            prec = cla_score_table[dt]['found_pred'] / (cla_score_table[dt]['found_pred'] + cla_score_table[dt]['fakefound_pred'] + 1e-8)
            recall = cla_score_table[dt]['found_label'] / (cla_score_table[dt]['found_label'] + cla_score_table[dt]['nofound_label'] + 1e-8)
            f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
            cla_score_table[dt]['prec'] = prec
            cla_score_table[dt]['recall'] = recall
            cla_score_table[dt]['f1'] = f1

            for cls_id in DatasetReader.cla_class_ids:
                prec = cla_score_table[dt][cls_id]['found_pred'] / (cla_score_table[dt][cls_id]['found_pred'] + cla_score_table[dt][cls_id]['fakefound_pred'] + 1e-8)
                recall = cla_score_table[dt][cls_id]['found_label'] / (cla_score_table[dt][cls_id]['found_label'] + cla_score_table[dt][cls_id]['nofound_label'] + 1e-8)
                f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
                cla_score_table[dt][cls_id]['prec'] = prec
                cla_score_table[dt][cls_id]['recall'] = recall
                cla_score_table[dt][cls_id]['f1'] = f1

        yaml.dump(cla_score_table, open(os.path.join(out_dir, 'all_cla.txt'), 'w'))


if __name__ == '__main__':
    from main_net7 import MainNet
    main(MainNet)

