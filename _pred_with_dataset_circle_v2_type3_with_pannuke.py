import os
import torch
import torch.nn.functional as F
import time
import numpy as np
import yaml
from dataset_reader_PanNuke import DatasetReader
from tensorboardX import SummaryWriter
import cv2
import imageio
import copy
from my_py_lib import im_tool
from my_py_lib import contour_tool
from my_py_lib.auto_show_running import AutoShowRunning

from a_config import project_root, device, net_in_hw, net_out_hw, batch_size, epoch, batch_count, eval_which_checkpoint,\
    dataset_path, process_control, net_save_dir, match_distance_thresh_list, b1_is_ce_loss, b2_is_ce_loss, b3_is_ce_loss, make_cla_is_det
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


use_heatmap_nms = True
use_single_pair = True
use_step4_process = False
# pred_heatmap_thresh_list = (0.2, 0.3, 0.4, 0.5, 0.6)
pred_heatmap_thresh_list = (0.4,)

if use_step4_process:
    process_control = [50, 100, 200]

in_dim = 3
# device = torch.device(0)
# net_in_hw = (320, 320)
# net_out_hw = (160, 160)
# batch_size = 3
# epoch = 1000
# batch_count = 5
auto_show_interval = 4

cla_class_num = DatasetReader.class_num - 1
cla_class_ids = list(range(cla_class_num))
cla_class_names = DatasetReader.class_names[:-1]


pred_train_out_dir = 'cla_pn_pred_circle_train_out_dir_tdss'
pred_valid_out_dir = 'cla_pn_pred_circle_valid_out_dir_tdss'
pred_test_out_dir = 'cla_pn_pred_circle_test_out_dir_tdss'


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


get_pg_id = eval_utils.get_pg_id
get_pg_name = eval_utils.get_pg_name


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

    start_epoch = 80

    pg_id = get_pg_id(start_epoch, process_control)

    # train_dataset = DatasetReader(os.path.join(project_root, 'pannuke_packed_dataset'), 'fold_1', pt_radius_min=1, pt_radius_max=1, rescale=0.5, use_repel_code=False, include_cls=None)
    # valid_dataset = DatasetReader(os.path.join(project_root, 'pannuke_packed_dataset'), 'fold_2', pt_radius_min=1, pt_radius_max=1, rescale=0.5, use_repel_code=False, include_cls=None)
    # test_dataset = DatasetReader(os.path.join(project_root, 'pannuke_packed_dataset'), 'fold_3', pt_radius_min=1, pt_radius_max=1, rescale=0.5, use_repel_code=False, include_cls=None)
    train_dataset = DatasetReader("D:\\数据集\\fold1_2_3", 'fold_1', pt_radius_min=1, pt_radius_max=1, rescale=0.5, use_repel_code=False, include_cls=None)
    valid_dataset = DatasetReader("D:\\数据集\\fold1_2_3", 'fold_2', pt_radius_min=1, pt_radius_max=1, rescale=0.5, use_repel_code=False, include_cls=None)
    test_dataset = DatasetReader("D:\\数据集\\fold1_2_3", 'fold_3', pt_radius_min=1, pt_radius_max=1, rescale=0.5, use_repel_code=False, include_cls=None)

    # 定义网络
    b1_out_dim = 1+1 if b1_is_ce_loss else 1
    b2_out_dim = 1+1 if b2_is_ce_loss else 1
    b3_out_dim = cla_class_num+1 if b3_is_ce_loss else cla_class_num
    net = NetClass(3, b1_out_dim, b2_out_dim, b3_out_dim)

    net.enabled_b2_branch = True
    net.enabled_b3_branch = True

    if eval_which_checkpoint == 'last':
        print('Will load last weight.')
        new_ck_name = get_pg_name(ck_name, start_epoch, process_control)
        net.load_state_dict(torch.load(new_ck_name, 'cpu'))
        print('load model success')
    elif eval_which_checkpoint == 'best':
        print('Will load best weight.')
        new_ck_name = get_pg_name(ck_best_name, start_epoch, process_control)
        net.load_state_dict(torch.load(new_ck_name, 'cpu'))
        print('load model success')
    # elif eval_which_checkpoint == 'minloss':
    #     print('Will load minloss weight.')
    #     new_ck_name = get_pg_name(ck_minloss_name, start_epoch)
    #     net.load_state_dict(torch.load(new_ck_name, 'cpu'))
    #     print('load model success')
    else:
        print('Unknow weight type. Will not load weight.')

    net = net.to(device)
    net.eval()

    # 分割相关
    for heatmap_thresh in pred_heatmap_thresh_list:
        for did, cur_dataset in enumerate([train_dataset, valid_dataset, test_dataset]):
            out_dir = [pred_train_out_dir, pred_valid_out_dir, pred_test_out_dir][did]

            out_dir = '{}/{}'.format(out_dir, str(heatmap_thresh))

            # if did != 1:
            #     continue

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

                im, label, im_info = cur_dataset.get_item_point(pid, use_enhance=False, window_hw=None)

                im_basename = im_info['im_id']

                # label = np.max(label[..., :5], 3, keepdims=True)
                label: dict = im_info['label']
                label_det_pts = []
                group_label_cls_pts = {}
                for k, v in label.items():
                    # 类别5是不准确的，需要去除
                    if k == 5:
                        continue
                    label_det_pts.extend(v)
                    if k not in group_label_cls_pts:
                        group_label_cls_pts[k] = []
                    group_label_cls_pts[k].append(v)

                label_det_pts = np.array(label_det_pts)
                for c in group_label_cls_pts:
                    group_label_cls_pts[c] = np.reshape(group_label_cls_pts[c], [-1, 2])

                print('Processing {}'.format(im_basename))

                del label

                # # test B
                # tmp_show = './tmp_show'
                # os.makedirs(tmp_show, exist_ok=True)

                # 运行区
                wim = BigPicPatch(1+1+cla_class_num, [im], (0, 0), window_hw=net_in_hw, level_0_patch_hw=(1, 1), custom_patch_merge_pipe=eval_utils.patch_merge_func, patch_border_pad_value=255, ignore_patch_near_border_ratio=0.5)
                gen = wim.batch_get_im_patch_gen(batch_size * 3)
                for batch_info, batch_patch in gen:
                    batch_patch = torch.tensor(np.asarray(batch_patch), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.
                    batch_pred_det, batch_pred_det2, batch_pred_cla = net(batch_patch)

                    if b1_is_ce_loss:
                        batch_pred_det = batch_pred_det.softmax(1)[:, 1:]
                    else:
                        batch_pred_det = batch_pred_det.clamp(0, 1)

                    if b2_is_ce_loss:
                        batch_pred_det2 = batch_pred_det2.softmax(1)[:, 1:]
                    else:
                        batch_pred_det2 = batch_pred_det2.clamp(0, 1)

                    if b3_is_ce_loss:
                        batch_pred_cla = batch_pred_cla.softmax(1)[:, 1:]
                    else:
                        batch_pred_cla = batch_pred_cla.clamp(0, 1)

                    batch_pred = torch.cat([batch_pred_det, batch_pred_det2, batch_pred_cla], 1)
                    batch_pred = batch_pred.permute(0, 2, 3, 1).cpu().numpy()

                    # a1 = time.time()
                    wim.batch_update_result(batch_info, batch_pred)

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

                pred_pm = wim.multi_scale_result[0].data / np.clip(wim.multi_scale_mask[0].data, 1e-8, None)
                pred_det_rough_pm, pred_det_fine_pm, pred_cla_pm = np.split(pred_pm, [1, 2], -1)
                pred_det_final_pm = pred_det_rough_pm * pred_det_fine_pm
                pred_cla_final_pm = pred_det_rough_pm * pred_cla_pm

                # det rough
                pred_det_rough_pts = eval_utils.get_pts_from_hm(pred_det_rough_pm, heatmap_thresh)

                mix_pic_det_a1 = eval_utils.draw_hm_circle(im, pred_det_rough_pts, label_det_pts, 6)

                det_info = calc_a_sample_info_points_each_class([pred_det_rough_pts, [0] * len(pred_det_rough_pts)], [label_det_pts, [0] * len(label_det_pts)], [0],
                                                                match_distance_thresh_list,
                                                                use_post_pro=False, use_single_pair=use_single_pair)

                for dt in match_distance_thresh_list:
                    for cls in [0]:
                        det_score_table[dt]['found_pred'] += det_info[cls][dt]['found_pred']
                        det_score_table[dt]['fakefound_pred'] += det_info[cls][dt]['fakefound_pred']
                        det_score_table[dt]['found_label'] += det_info[cls][dt]['found_label']
                        det_score_table[dt]['nofound_label'] += det_info[cls][dt]['nofound_label']
                        det_score_table[dt]['pred_repeat'] += det_info[cls][dt]['pred_repeat']
                        det_score_table[dt]['label_repeat'] += det_info[cls][dt]['label_repeat']

                imageio.imwrite(os.path.join(out_dir, '{}_1det_a1_m.png'.format(im_basename)), mix_pic_det_a1)
                imageio.imwrite(os.path.join(out_dir, '{}_1det_a1_h.png'.format(im_basename)), (pred_det_rough_pm * 255).astype(np.uint8))
                yaml.dump(det_info, open(os.path.join(out_dir, '{}_det.txt'.format(im_basename)), 'w'))

                # det fine

                if use_heatmap_nms:
                    pred_det_final_pm[..., 0] = pred_det_final_pm[..., 0] * heatmap_nms(pred_det_final_pm[..., 0])

                pred_det_post_pts = eval_utils.get_pts_from_hm(pred_det_final_pm, heatmap_thresh)

                mix_pic_det_a2 = eval_utils.draw_hm_circle(im, pred_det_post_pts, label_det_pts, 6)

                det_info_a2 = calc_a_sample_info_points_each_class([pred_det_post_pts, [0] * len(pred_det_post_pts)],
                                                                   [label_det_pts, [0] * len(label_det_pts)], [0],
                                                                   match_distance_thresh_list, use_single_pair=use_single_pair)

                for dt in match_distance_thresh_list:
                    for cls in [0]:
                        det_score_table_a2[dt]['found_pred'] += det_info_a2[cls][dt]['found_pred']
                        det_score_table_a2[dt]['fakefound_pred'] += det_info_a2[cls][dt]['fakefound_pred']
                        det_score_table_a2[dt]['found_label'] += det_info_a2[cls][dt]['found_label']
                        det_score_table_a2[dt]['nofound_label'] += det_info_a2[cls][dt]['nofound_label']
                        det_score_table_a2[dt]['pred_repeat'] += det_info_a2[cls][dt]['pred_repeat']
                        det_score_table_a2[dt]['label_repeat'] += det_info_a2[cls][dt]['label_repeat']

                # cla
                pred_cla_final_pm_hm_before = np.copy(pred_cla_final_pm)
                if use_heatmap_nms:
                    for c in range(pred_cla_final_pm.shape[2]):
                        pred_cla_final_pm[:, :, c] = pred_cla_final_pm[:, :, c] * heatmap_nms(pred_cla_final_pm[:, :, c])

                ## 老方法，从cla中获得
                # pred_cla_pts = []
                # pred_cla_pts_cla = []
                # for i in range(pred_cla_final_pm.shape[2]):
                #     pts = eval_utils.get_pts_from_hm(pred_cla_final_pm[..., i:i+1], heatmap_thresh)
                #     pred_cla_pts.extend(pts)
                #     pred_cla_pts_cla.extend([i] * len(pts))

                ## 新方法，从det中获得
                pred_cla_pts = []
                pred_cla_pts_cla = []
                pred_cla_pts.extend(pred_det_post_pts)
                pred_cla_pts_cla.extend(eval_utils.get_cls_pts_from_hm(pred_cla_pts, pred_cla_final_pm_hm_before))

                label_cla_pts = []
                label_cla_pts_cla = []
                for i in group_label_cls_pts:
                    label_cla_pts.extend(group_label_cls_pts[i])
                    label_cla_pts_cla.extend([i] * len(group_label_cls_pts[i]))

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

                imageio.imwrite(os.path.join(out_dir, '{}_1det_a2_m.png'.format(im_basename)), mix_pic_det_a2)
                imageio.imwrite(os.path.join(out_dir, '{}_1det_a2_h_1before.png'.format(im_basename)), (pred_det_fine_pm * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(out_dir, '{}_1det_a2_h_2after.png'.format(im_basename)), (pred_det_final_pm * 255).astype(np.uint8))
                yaml.dump(det_info_a2, open(os.path.join(out_dir, '{}_det_a2.txt'.format(im_basename)), 'w'))

                # imageio.imwrite(os.path.join(out_dir, '{}_1det_a3.png'.format(im_basename)), mix_pic_det_a3)

                # 输出分类分支的图
                for i in range(cla_class_num):
                    imageio.imwrite(os.path.join(out_dir, '{}_2cla_b1_{}.png'.format(im_basename, cla_class_names[i])), group_mix_pic_cla[i])

                    # 生成分类分支的处理前和处理后的热图
                    imageio.imwrite(os.path.join(out_dir, '{}_2cla_b2_{}_1before.png'.format(im_basename, cla_class_names[i])),
                                    (pred_cla_pm[..., i] * 255).astype(np.uint8))
                    imageio.imwrite(os.path.join(out_dir, '{}_2cla_b2_{}_2after.png'.format(im_basename, cla_class_names[i])),
                                    (pred_cla_final_pm[..., i] * 255).astype(np.uint8))

                # bug check start
                det_label_cnt = 0
                det_label_cnt += det_info_a2[0][6]['found_label'] + det_info_a2[0][6]['nofound_label']
                cla_label_cnt = 0
                for cls in range(5):
                    cla_label_cnt += cla_info[cls][6]['found_label'] + cla_info[cls][6]['nofound_label']

                if det_label_cnt != cla_label_cnt:
                    print('PID', pid)
                    raise AssertionError('Error! Bad label cnt.')
                # bug check end


            # 计算det a1 F1，精确率，召回率
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

                for cls_id in cla_class_ids:
                    prec = cla_score_table[dt][cls_id]['found_pred'] / (
                                cla_score_table[dt][cls_id]['found_pred'] + cla_score_table[dt][cls_id]['fakefound_pred'] + 1e-8)
                    recall = cla_score_table[dt][cls_id]['found_label'] / (
                                cla_score_table[dt][cls_id]['found_label'] + cla_score_table[dt][cls_id]['nofound_label'] + 1e-8)
                    f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
                    cla_score_table[dt][cls_id]['prec'] = prec
                    cla_score_table[dt][cls_id]['recall'] = recall
                    cla_score_table[dt][cls_id]['f1'] = f1

            yaml.dump(cla_score_table, open(os.path.join(out_dir, 'all_cla.txt'), 'w'))


if __name__ == '__main__':
    from main_net11 import MainNet
    main(MainNet)

