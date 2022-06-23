import os
import torch
import torch.nn.functional as F
import time
import numpy as np
import yaml
from dataset_reader2 import DatasetReader
from tensorboardX import SummaryWriter
import cv2
# import prettytable
import uuid
import loss_func

from a_config import project_root, device, net_in_hw, net_out_hw, batch_size, epoch, batch_count,\
    is_train_from_recent_checkpoint, dataset_path, net_save_dir, net_train_logs_dir,\
    b1_is_ce_loss, b1_loss_type, b2_is_ce_loss, b2_loss_type, train_lr,\
    match_distance_thresh_list, process_control, make_cla_is_det, use_repel_code
from my_py_lib.preload_generator import preload_generator
import eval_utils
import visdom
import math
import big_pic_result
import heatmap_nms


'''
注意：破坏性更改，第一阶段使用ce_loss，第二阶段使用L2。
记得标注恢复点
现在开始兼容破坏性更改，支持不同类loss
'''
get_pg_name = eval_utils.get_pg_name
get_pg_id = eval_utils.get_pg_id

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

save_dir = net_save_dir
log_dir = net_train_logs_dir

os.makedirs(save_dir, exist_ok=True)

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


def tr_pm_to_onehot_vec(label_pm: torch.Tensor, soft_label=True):
    if soft_label:
        label_oh = label_pm
    else:
        label_oh = (label_pm > 0.5).type(torch.float32)

    bg = 1 - label_oh.max(1, keepdim=True)[0]
    label_oh = torch.cat([bg, label_oh], dim=1)
    return label_oh


def tr_cla_vec_to_det_vec(label_cla_ce: torch.Tensor):
    bg = (label_cla_ce[:, 0:1] > 0.5).type(torch.float32)
    pos = 1 - bg
    det_vec = torch.cat([bg, pos], dim=1)
    return det_vec


def main(NetClass):
    _last_auto_show_update_time = 0

    model_id = NetClass.model_id

    single_id = str(uuid.uuid1())
    print('single_id:', single_id)
    vtb = visdom.Visdom(env=single_id)

    ck_name         = '{}/{}_model.pt'          .format(save_dir, model_id)
    ck_best_name    = '{}/{}_model_best.pt'     .format(save_dir, model_id)
    # ck_minloss_name = '{}/{}_model_minloss.pt'  .format(save_dir, model_id)
    ck_optim_name   = '{}/{}_optim.pt'          .format(save_dir, model_id)
    ck_restart_name = '{}/{}_restart.yml'       .format(save_dir, model_id)
    # ck_best_extra_name      = '{}/{}_best_extra.txt'    .format(save_dir, model_id)
    # ck_minloss_extra_name   = '{}/{}_minloss_extra.txt' .format(save_dir, model_id)
    score_det_name      = '{}/{}_score_det.txt' .format(save_dir, model_id)
    score_cla_name      = '{}/{}_score_cla.txt' .format(save_dir, model_id)
    score_det_best_name = '{}/{}_score_det_best.txt'.format(save_dir, model_id)
    score_cla_best_name = '{}/{}_score_cla_best.txt'.format(save_dir, model_id)
    # score_det_minloss_name = '{}/{}_score_det_minloss.txt'.format(save_dir, model_id)
    # score_cla_minloss_name = '{}/{}_score_cla_minloss.txt'.format(save_dir, model_id)

    logdir = '{}_{}' .format(log_dir, model_id)
    sw = SummaryWriter(logdir)

    train_dataset = DatasetReader(dataset_path, 'train', pt_radius_min=9, pt_radius_max=9, use_repel_code=use_repel_code)
    eval_dataset = DatasetReader(dataset_path, 'valid', pt_radius_min=9, pt_radius_max=9, use_repel_code=False)

    # 定义网络
    b1_out_dim = 1+1 if b1_is_ce_loss else 1
    b2_out_dim = cla_class_num+1 if b2_is_ce_loss else cla_class_num
    net = NetClass(3, b1_out_dim, b2_out_dim)

    net = net.to(device)

    optim = torch.optim.Adam(net.parameters(), train_lr, eps=1e-8, weight_decay=1e-6)
    optim_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 10, 1e-6)

    start_epoch = 0
    pg_max_valid_value = [-math.inf for _ in range(len(process_control)+1)]
    pg_minloss_value = [math.inf for _ in range(len(process_control)+1)]

    if is_train_from_recent_checkpoint and os.path.isfile(ck_restart_name):
        d = yaml.safe_load(open(ck_restart_name, 'r'))
        start_epoch = d['start_epoch']
        pg_max_valid_value = d['pg_max_valid_value']
        pg_minloss_value = d['pg_minloss_value']

        if start_epoch not in process_control:
            new_ck_name = get_pg_name(ck_name, start_epoch, process_control)
            new_ck_optim_name = get_pg_name(ck_optim_name, start_epoch)
            net.load_state_dict(torch.load(new_ck_name, 'cpu'))
            optim.load_state_dict(torch.load(new_ck_optim_name, 'cpu'))

    # 用来快速跳过循环，快速检查代码问题
    use_one = False
    _do_one = False

    # 过程id

    for e in range(start_epoch, epoch):
        optim_adjust.step(e)
        net.train()
        train_det_acc = 0
        train_cla_acc = 0
        train_loss = 0

        # 每次变更阶段，重载上阶段的best权重
        if e in process_control:
            new_ck_name = get_pg_name(ck_best_name, e - 1, process_control)
            new_ck_optim_name = get_pg_name(ck_optim_name, e - 1, process_control)
            net.load_state_dict(torch.load(new_ck_name, 'cpu'))
            optim.load_state_dict(torch.load(new_ck_optim_name, 'cpu'))

        pg_id = get_pg_id(e, process_control)

        if pg_id == 1:
            net.enabled_cls_branch = False
            net.set_freeze_seg1(False)
            net.set_freeze_seg2(True)
        elif pg_id == 2:
            net.enabled_cls_branch = True
            net.set_freeze_seg1(True)
            net.set_freeze_seg2(False)
        elif pg_id == 3:
            net.enabled_cls_branch = True
            net.set_freeze_seg1(False)
            net.set_freeze_seg2(False)
        else:
            raise AssertionError()

        train_gen = train_dataset.get_train_batch_gen(batch_count, batch_size, use_enhance=True, window_hw=net_in_hw)
        train_gen = preload_generator(train_gen)

        for b in range(batch_count):
            ori_batch_im, ori_batch_label, ori_batch_ignore_mask, ori_batch_info = next(train_gen)

            ori_batch_label_det, ori_batch_label_cla = np.split(ori_batch_label, [1,], 3)

            if make_cla_is_det:
                ori_batch_label_cla = ori_batch_label_det

            batch_im = torch.tensor(ori_batch_im, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255
            batch_label_det = torch.tensor(ori_batch_label_det, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            if make_cla_is_det:
                batch_label_cla = batch_label_det
                batch_label_mask = torch.ones([batch_label_det.shape[0], 1, batch_label_det.shape[2], batch_label_det.shape[3]], dtype=torch.float32)
            else:
                batch_label_cla = torch.tensor(ori_batch_label_cla, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                batch_label_mask = 1 - torch.tensor(ori_batch_ignore_mask, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

            batch_label_det_oh = tr_pm_to_onehot_vec(batch_label_det, soft_label=False)
            batch_label_cla_oh = tr_pm_to_onehot_vec(batch_label_cla, soft_label=True)

            batch_patch_pred_det, batch_patch_pred_cla = net(batch_im)

            # 训练时检查像素准确率，这里的准确率按照类别平均检查
            with torch.no_grad():
                # batch_pred_cm = torch.argmax(batch_pred_pm, 1, keepdim=False).cpu()
                batch_pred_det_tmp = batch_patch_pred_det
                if b1_is_ce_loss:
                    batch_pred_det_tmp = (batch_pred_det_tmp.softmax(1)[:, 1:] > 0.5).type(torch.float32)
                else:
                    batch_pred_det_tmp = (batch_pred_det_tmp.clamp(0., 1.) > 0.5).type(torch.float32)

                det_acc = (batch_pred_det_tmp * batch_label_det).sum() / (batch_label_det.sum() + 1e-8).item()

                if pg_id > 1:
                    batch_pred_cla_tmp = batch_patch_pred_cla
                    if b2_is_ce_loss:
                        batch_pred_cla_tmp = (batch_pred_cla_tmp.softmax(1)[:, 1:] > 0.5).type(torch.float32)
                    else:
                        batch_pred_cla_tmp = (batch_pred_cla_tmp.clamp(0., 1.) > 0.5).type(torch.float32)
                    cla_acc = ((batch_pred_cla_tmp * batch_label_cla).sum(dim=(0, 2, 3)) / (batch_label_cla.sum(dim=(0, 2, 3)) + 1e-8)).mean().item()
                else:
                    cla_acc = 0.

                if time.time() - _last_auto_show_update_time > 60:
                    _last_auto_show_update_time = time.time()

                    vtb.images(batch_im, nrow=nrow, win='ori_ims', opts={'title': 'ori_ims'})

                    show_ims(vtb, batch_label_det.mul(255).type(torch.uint8), 'label_det')
                    show_ims(vtb, batch_label_det_oh.mul(255).type(torch.uint8), 'label_det_oh')
                    show_ims(vtb, batch_label_cla.mul(255).type(torch.uint8), 'label_cla')
                    show_ims(vtb, batch_pred_det_tmp.mul(255).type(torch.uint8), 'pred_det')
                    show_ims(vtb, (batch_pred_det_tmp - batch_label_det).abs().mul(255).type(torch.uint8), 'det_error')
                    if pg_id > 1:
                        show_ims(vtb, batch_pred_cla_tmp.mul(255).type(torch.uint8), 'pred_cla')
                        show_ims(vtb, (batch_label_det * batch_pred_cla_tmp).mul(255).type(torch.uint8), 'pred_cla_final')
                        show_ims(vtb, batch_label_mask.mul(255).type(torch.uint8), 'ignore_mask')

            if pg_id == 1:
                assert b1_is_ce_loss
                loss = loss_func.det_loss(batch_patch_pred_det.softmax(dim=1), batch_label_det_oh, 'auto')
            elif pg_id == 2:
                assert not b2_is_ce_loss
                loss = loss_func.a_cla_loss(batch_patch_pred_det.softmax(dim=1)[:, 1:2], batch_patch_pred_cla, batch_label_cla, 'auto')
            elif pg_id == 3:
                loss1 = loss_func.det_loss(batch_patch_pred_det.softmax(dim=1), batch_label_det_oh, 'auto')
                loss2 = loss_func.a_cla_loss(batch_patch_pred_det.softmax(dim=1)[:, 1:2], batch_patch_pred_cla, batch_label_cla, 'auto')
                loss = loss1 + loss2
            else:
                raise AssertionError('Unknow process_step')

            train_det_acc += det_acc
            train_cla_acc += cla_acc
            train_loss += loss.item()

            print('epoch: {} count: {} train det acc: {:.3f} train cla acc: {:.3f} loss: {:.3f}'.format(e, b, det_acc, cla_acc, loss.item()))
            optim.zero_grad()
            assert not np.isnan(loss.item()), 'Found loss Nan!'
            loss.backward()
            optim.step()

        train_det_acc = train_det_acc / batch_count
        train_cla_acc = train_cla_acc / batch_count
        train_loss = train_loss / batch_count

        sw.add_scalar('train_det_acc', train_det_acc, global_step=e)
        sw.add_scalar('train_cla_acc', train_cla_acc, global_step=e)
        sw.add_scalar('train_loss', train_loss, global_step=e)

        # here to check eval
        if (e+1) % 1 == 0:
            with torch.no_grad():
                net.eval()
                net.enabled_cls_branch = True

                det_score_table = {'epoch': e,
                                   'det_pix_pred_found': 0,
                                   'det_pix_pred_fakefound': 0,
                                   'det_pix_label_found': 0,
                                   'det_pix_label_nofound': 0,
                                   'det_pix_recall': 0,
                                   'det_pix_prec': 0,
                                   'det_pix_f1': 0,
                                   'det_pix_f2': 0,
                                   }
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

                cla_score_table = {'epoch': e}
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

                patch_count = 0
                eval_loss = 0.

                for t_i in range(len(eval_dataset)):
                    patch_count += 1

                    im, label, ignore_mask, info = eval_dataset.get_item(t_i, use_enhance=False, window_hw=None)

                    label_det, label_cla = np.split(label, [1,], -1)

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

                    wim = big_pic_result.BigPicPatch(1+cla_class_num, [im], (0, 0), net_in_hw, (0, 0), 0, 0, custom_patch_merge_pipe=eval_utils.patch_merge_func, patch_border_pad_value=255)

                    gen = wim.batch_get_im_patch_gen(batch_size)
                    gen = preload_generator(gen)
                    for batch_info, batch_patch in gen:
                        batch_patch = torch.tensor(np.asarray(batch_patch), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.
                        batch_patch_pred_det, batch_patch_pred_cla = net(batch_patch)

                        if b1_is_ce_loss:
                            batch_patch_pred_det = batch_patch_pred_det.softmax(1)[:, 1:]
                        else:
                            batch_patch_pred_det = batch_patch_pred_det.clamp(0, 1)

                        if b2_is_ce_loss:
                            batch_patch_pred_cla = batch_patch_pred_cla.softmax(1)[:, 1:]
                        else:
                            batch_patch_pred_cla = batch_patch_pred_cla.clamp(0, 1)

                        batch_pred = torch.cat([batch_patch_pred_det, batch_patch_pred_cla], 1)
                        batch_pred = batch_pred.permute(0, 2, 3, 1).cpu().numpy()

                        wim.batch_update_result(batch_info, batch_pred)

                    pred_pm = wim.multi_scale_result[0].data / np.clip(wim.multi_scale_mask[0].data, 1e-8, None)
                    pred_det_pm, pred_cla_pm = np.split(pred_pm, [1,], -1)
                    pred_cla_final_pm = pred_det_pm * pred_cla_pm
                    pred_cla_final_with_ignore_pm = pred_cla_final_pm * (1 - ignore_mask)

                    # 统计像素分类数据
                    det_pix_pred_bin = (pred_pm[..., 0:1] > 0.5).astype(dtype=np.float32)
                    det_pix_label_bin = (label[..., 0:1] > 0.5).astype(dtype=np.float32)

                    det_score_table['det_pix_pred_found'] += float((det_pix_pred_bin * det_pix_label_bin).sum(dtype=np.float32))
                    det_score_table['det_pix_pred_fakefound'] += float((det_pix_pred_bin * (1 - det_pix_label_bin)).sum(dtype=np.float32))
                    det_score_table['det_pix_label_found'] += det_score_table['det_pix_pred_found']
                    det_score_table['det_pix_label_nofound'] += float(((1 - det_pix_pred_bin) * det_pix_label_bin).sum(dtype=np.float32))

                    # 统计点找到数据
                    det_info = eval_utils.calc_a_sample_info_points_each_class(pred_det_pm, [label_det_pts, [0]*len(label_det_pts)], [0], match_distance_thresh_list,
                                                                            use_post_pro=False, use_single_pair=True)

                    tmp_label_cla_without_ignore_pos = []
                    tmp_label_cla_without_ignore_cla = []
                    for k in group_label_cla_pts_without_ignore:
                        tmp_label_cla_without_ignore_pos.extend(group_label_cla_pts_without_ignore[k])
                        if make_cla_is_det:
                            tmp_label_cla_without_ignore_cla.extend([k] * len(group_label_cla_pts_without_ignore[k]))
                        else:
                            tmp_label_cla_without_ignore_cla.extend([k-1] * len(group_label_cla_pts_without_ignore[k]))

                    if pg_id > 1:
                        cla_info = eval_utils.calc_a_sample_info_points_each_class(pred_cla_final_with_ignore_pm,
                                [tmp_label_cla_without_ignore_pos, tmp_label_cla_without_ignore_cla], cla_class_ids, match_distance_thresh_list,
                                use_post_pro=False, use_single_pair=True)
                    else:
                        cla_info = None

                    for dt in match_distance_thresh_list:
                        for cls in [0]:
                            det_score_table[dt]['found_pred'] += det_info[cls][dt]['found_pred']
                            det_score_table[dt]['fakefound_pred'] += det_info[cls][dt]['fakefound_pred']
                            det_score_table[dt]['found_label'] += det_info[cls][dt]['found_label']
                            det_score_table[dt]['nofound_label'] += det_info[cls][dt]['nofound_label']
                            det_score_table[dt]['pred_repeat'] += det_info[cls][dt]['pred_repeat']
                            det_score_table[dt]['label_repeat'] += det_info[cls][dt]['label_repeat']

                    if cla_info is not None:
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

                # 计算det的像素的F1，精确率，召回率
                det_score_table['det_pix_prec'] = det_score_table['det_pix_pred_found'] / (
                            det_score_table['det_pix_pred_fakefound'] + det_score_table['det_pix_pred_found'] + 1e-8)
                det_score_table['det_pix_recall'] = det_score_table['det_pix_label_found'] / (
                            det_score_table['det_pix_label_nofound'] + det_score_table['det_pix_label_found'] + 1e-8)
                det_score_table['det_pix_f1'] = 2 * det_score_table['det_pix_prec'] * det_score_table['det_pix_recall'] / (
                            det_score_table['det_pix_prec'] + det_score_table['det_pix_recall'] + 1e-8)
                det_score_table['det_pix_f2'] = 5 * det_score_table['det_pix_prec'] * det_score_table['det_pix_recall'] / (
                            det_score_table['det_pix_prec'] * 4 + det_score_table['det_pix_recall'] + 1e-8)

                # 计算det的F1，精确率，召回率
                for dt in match_distance_thresh_list:
                    prec = det_score_table[dt]['found_pred'] / (det_score_table[dt]['found_pred'] + det_score_table[dt]['fakefound_pred'] + 1e-8)
                    recall = det_score_table[dt]['found_label'] / (det_score_table[dt]['found_label'] + det_score_table[dt]['nofound_label'] + 1e-8)
                    f1 = 2 * (prec * recall) / (prec + recall + 1e-8)
                    det_score_table[dt]['prec'] = prec
                    det_score_table[dt]['recall'] = recall
                    det_score_table[dt]['f1'] = f1

                # 计算cla的F1，精确率，召回率
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

                # eval_loss = eval_loss / patch_count

                for dt in match_distance_thresh_list:
                    for cls in [0]:
                        sw.add_scalar('{}/{}/found_pred'.format(dt, cls), det_score_table[dt]['found_pred'], global_step=e)
                        sw.add_scalar('{}/{}/fakefound_pred'.format(dt, cls), det_score_table[dt]['fakefound_pred'], global_step=e)
                        sw.add_scalar('{}/{}/found_label'.format(dt, cls), det_score_table[dt]['found_label'], global_step=e)
                        sw.add_scalar('{}/{}/nofound_label'.format(dt, cls), det_score_table[dt]['nofound_label'], global_step=e)
                        sw.add_scalar('{}/{}/pred_repeat'.format(dt, cls), det_score_table[dt]['pred_repeat'], global_step=e)
                        sw.add_scalar('{}/{}/label_repeat'.format(dt, cls), det_score_table[dt]['label_repeat'], global_step=e)
                        sw.add_scalar('{}/{}/prec'.format(dt, cls), det_score_table[dt]['prec'], global_step=e)
                        sw.add_scalar('{}/{}/recall'.format(dt, cls), det_score_table[dt]['recall'], global_step=e)
                        sw.add_scalar('{}/{}/f1'.format(dt, cls), det_score_table[dt]['f1'], global_step=e)

                for dt in match_distance_thresh_list:
                    for cls in cla_class_ids:
                        sw.add_scalar('{}/{}/found_pred'.format(dt, cls), cla_score_table[dt][cls]['found_pred'], global_step=e)
                        sw.add_scalar('{}/{}/fakefound_pred'.format(dt, cls), cla_score_table[dt][cls]['fakefound_pred'], global_step=e)
                        sw.add_scalar('{}/{}/found_label'.format(dt, cls), cla_score_table[dt][cls]['found_label'], global_step=e)
                        sw.add_scalar('{}/{}/nofound_label'.format(dt, cls), cla_score_table[dt][cls]['nofound_label'], global_step=e)
                        sw.add_scalar('{}/{}/pred_repeat'.format(dt, cls), cla_score_table[dt][cls]['pred_repeat'], global_step=e)
                        sw.add_scalar('{}/{}/label_repeat'.format(dt, cls), cla_score_table[dt][cls]['label_repeat'], global_step=e)
                        sw.add_scalar('{}/{}/prec'.format(dt, cls), cla_score_table[dt][cls]['prec'], global_step=e)
                        sw.add_scalar('{}/{}/recall'.format(dt, cls), cla_score_table[dt][cls]['recall'], global_step=e)
                        sw.add_scalar('{}/{}/f1'.format(dt, cls), cla_score_table[dt][cls]['f1'], global_step=e)

                # 打印评分
                out_line = yaml.safe_dump(det_score_table) + '\n' + yaml.safe_dump(cla_score_table)

                print('epoch {}'.format(e), out_line)

                if pg_id == 1:
                    # current_valid_value = det_score_table[match_distance_thresh_list[0]]['f1']
                    current_valid_value = det_score_table['det_pix_f2']
                else:
                    current_valid_value = cla_score_table[match_distance_thresh_list[0]]['f1']

                # current_eval_loss = eval_loss

                # 保存最好的
                if current_valid_value > pg_max_valid_value[pg_id - 1]:
                    pg_max_valid_value[pg_id - 1] = current_valid_value
                    new_ck_best_name = get_pg_name(ck_best_name, e, process_control)
                    torch.save(net.state_dict(), new_ck_best_name)
                    # new_ck_best_extra_name = get_pg_name(ck_best_extra_name, e)
                    # open(new_ck_best_extra_name, 'w').write(out_line)
                    new_score_det_best_name = get_pg_name(score_det_best_name, e, process_control)
                    new_score_cla_best_name = get_pg_name(score_cla_best_name, e, process_control)
                    yaml.safe_dump(det_score_table, open(new_score_det_best_name, 'w'))
                    yaml.safe_dump(cla_score_table, open(new_score_cla_best_name, 'w'))

                # if current_eval_loss < pg_minloss_value[pg_id - 1]:
                #     pg_minloss_value[pg_id - 1] = current_eval_loss
                #     new_ck_minloss_name = get_pg_name(ck_minloss_name, e)
                #     torch.save(net.state_dict(), new_ck_minloss_name)
                #     # new_ck_minloss_extra_name = get_pg_name(ck_minloss_extra_name, e)
                #     # open(new_ck_minloss_extra_name, 'w').write(out_line)
                #     new_score_det_minloss_name = get_pg_name(score_det_minloss_name, e)
                #     new_score_cla_minloss_name = get_pg_name(score_cla_minloss_name, e)
                #     yaml.safe_dump(det_score_table, open(new_score_det_minloss_name, 'w'))
                #     yaml.safe_dump(cla_score_table, open(new_score_cla_minloss_name, 'w'))

                new_ck_name = get_pg_name(ck_name, e, process_control)
                new_ck_optim_name = get_pg_name(ck_optim_name, e, process_control)
                torch.save(net.state_dict(), new_ck_name)
                torch.save(optim.state_dict(), new_ck_optim_name)
                d = {'start_epoch': e + 1, 'pg_max_valid_value': pg_max_valid_value,
                     'pg_minloss_value': pg_minloss_value}
                yaml.safe_dump(d, open(ck_restart_name, 'w'))
                new_score_det_name = get_pg_name(score_det_name, e, process_control)
                new_score_cla_name = get_pg_name(score_cla_name, e, process_control)
                yaml.safe_dump(det_score_table, open(new_score_det_name, 'w'))
                yaml.safe_dump(cla_score_table, open(new_score_cla_name, 'w'))

    sw.close()


if __name__ == '__main__':
    from main_net7 import MainNet
    main(MainNet)

