import os
import torch
import torch.nn.functional as F
import time
import numpy as np
import yaml
import cv2
import imageio
import copy

from a_config import project_root, device, net_in_hw, net_out_hw, batch_size, eval_which_checkpoint,\
    process_control, net_save_dir, match_distance_thresh_list, b1_is_ce_loss, b2_is_ce_loss, b3_is_ce_loss
import eval_utils
from big_pic_result import BigPicPatch
from heatmap_nms import heatmap_nms
import pickle


use_heatmap_nms = True
use_single_pair = True
# pred_heatmap_thresh_list = (0.2, 0.3, 0.4, 0.5, 0.6)
# pred_heatmap_thresh_list = (0.2, 0.3, 0.4, 0.5)
pred_heatmap_thresh_list = (0.3,)


get_pg_id = eval_utils.get_pg_id
get_pg_name = eval_utils.get_pg_name


in_dim = 3
# device = torch.device(0)
# net_in_hw = (320, 320)
# net_out_hw = (160, 160)
# batch_size = 3
# epoch = 1000
# batch_count = 5
auto_show_interval = 4

cla_class_num = 1
cla_class_ids = list(range(cla_class_num))
cla_class_names = ['det']


pred_out_dir = 'cla_ex_pred_circle_out_dir'

ex_dataset_path = project_root + '/extern_datasets'
ex_datasets = [
    'set1.pkl',
    'set2.pkl',
    'set3.pkl',
    'set4.pkl',
]

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

    # 定义网络
    b1_out_dim = 1+1 if b1_is_ce_loss else 1
    b2_out_dim = 1+1 if b2_is_ce_loss else 1
    b3_out_dim = 5+1 if b3_is_ce_loss else 5
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
        for _, cur_dataset_name in enumerate(ex_datasets):
            ds_basename = os.path.basename(cur_dataset_name)
            dataset_path = f'{ex_dataset_path}/{cur_dataset_name}'

            out_dir = f'{pred_out_dir}/{ds_basename}'
            out_dir = f'{out_dir}/{str(heatmap_thresh)}'
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

            cur_dataset = pickle.load(open(dataset_path, 'rb'))

            for item in cur_dataset:
                im_code = item['im_code']
                im_basename = item['basename']
                label = item['label']

                im = cv2.imdecode(im_code, -1)
                label = np.asarray(label, np.float32).reshape(-1, 2)

                # 原图为40x，要下采样为20x
                # 原始坐标格式是xy，要转换为yx格式
                im = im[..., :3]
                im = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2), interpolation=cv2.INTER_AREA)
                label = label[:, ::-1]
                label = label / 2

                label_det_pts = label

                print('Processing {}'.format(im_basename))

                del label

                # # test B
                # tmp_show = './tmp_show'
                # os.makedirs(tmp_show, exist_ok=True)

                # 运行区
                wim = BigPicPatch(1+1, [im], (0, 0), window_hw=net_in_hw, level_0_patch_hw=(1, 1), custom_patch_merge_pipe=eval_utils.patch_merge_func, patch_border_pad_value=255, ignore_patch_near_border_ratio=0.5)
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

                    # if b3_is_ce_loss:
                    #     batch_pred_cla = batch_pred_cla.softmax(1)[:, 1:]
                    # else:
                    #     batch_pred_cla = batch_pred_cla.clamp(0, 1)

                    batch_pred = torch.cat([batch_pred_det, batch_pred_det2], 1)
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

                pred_pm = wim.multi_scale_result[0].data / np.clip(wim.multi_scale_mask[0].data, 1e-8, None).astype(np.float32)
                pred_det_rough_pm, pred_det_fine_pm = np.split(pred_pm, [1,], -1)
                pred_det_final_pm = pred_det_rough_pm * pred_det_fine_pm
                # pred_cla_final_pm = pred_det_rough_pm * pred_cla_pm

                # det rough
                pred_det_rough_pts = eval_utils.get_pts_from_hm(pred_det_rough_pm, heatmap_thresh)

                mix_pic_det_a1 = eval_utils.draw_hm_circle(im, pred_det_rough_pts, label_det_pts, 6)

                det_info = eval_utils.calc_a_sample_info_points_each_class([pred_det_rough_pts, [0] * len(pred_det_rough_pts)], [label_det_pts, [0] * len(label_det_pts)], [0],
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

                det_info_a2 = eval_utils.calc_a_sample_info_points_each_class([pred_det_post_pts, [0] * len(pred_det_post_pts)],
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

                imageio.imwrite(os.path.join(out_dir, '{}_1det_a2_m.png'.format(im_basename)), mix_pic_det_a2)
                imageio.imwrite(os.path.join(out_dir, '{}_1det_a2_h_1before.png'.format(im_basename)), (pred_det_fine_pm * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(out_dir, '{}_1det_a2_h_2after.png'.format(im_basename)), (pred_det_final_pm * 255).astype(np.uint8))
                yaml.dump(det_info_a2, open(os.path.join(out_dir, '{}_det_a2.txt'.format(im_basename)), 'w'))

                # imageio.imwrite(os.path.join(out_dir, '{}_1det_a3.png'.format(im_basename)), mix_pic_det_a3)

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


if __name__ == '__main__':
    from main_net11 import MainNet
    main(MainNet)

