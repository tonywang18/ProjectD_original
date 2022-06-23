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
    is_train_from_recent_checkpoint, dataset_path, net_save_dir, net_train_logs_dir, train_lr,\
    match_distance_thresh_list, process_control, make_cla_is_det, use_repel_code
from my_py_lib.preload_generator import preload_generator
import eval_utils
import visdom
import math


'''
单纯性训练分类模型，并且使用在label 9个像素外不计算Loss
'''


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


def main(NetClass):
    _last_auto_show_update_time = 0

    model_id = NetClass.model_id

    single_id = str(uuid.uuid1())
    print('single_id:', single_id)
    vtb = visdom.Visdom(env=single_id)

    ck_name         = '{}/{}_model.pt'          .format(save_dir, model_id)
    ck_best_name    = '{}/{}_model_best.pt'     .format(save_dir, model_id)
    ck_minloss_name = '{}/{}_model_minloss.pt'  .format(save_dir, model_id)
    ck_optim_name   = '{}/{}_optim.pt'          .format(save_dir, model_id)
    ck_restart_name = '{}/{}_restart.yml'       .format(save_dir, model_id)
    score_cla_name          = '{}/{}_score_cla.txt' .format(save_dir, model_id)
    score_cla_best_name     = '{}/{}_score_cla_best.txt'.format(save_dir, model_id)
    score_cla_minloss_name  = '{}/{}_score_cla_minloss.txt'.format(save_dir, model_id)

    logdir = '{}_{}' .format(log_dir, model_id)
    sw = SummaryWriter(logdir)

    train_dataset = DatasetReader(dataset_path, 'train', pt_radius_min=9, pt_radius_max=9, use_repel_code=True, A=0.3)
    eval_dataset = DatasetReader(dataset_path, 'valid', pt_radius_min=9, pt_radius_max=9, use_repel_code=True, A=0.3)

    net = NetClass(3, cla_class_num)

    net = net.to(device)

    optim = torch.optim.Adam(net.parameters(), train_lr, eps=1e-8, weight_decay=1e-6)
    optim_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 20, 1e-6)

    start_epoch = 0
    max_valid_value = -math.inf
    minloss_value = math.inf

    if is_train_from_recent_checkpoint and os.path.isfile(ck_restart_name):
        d = yaml.safe_load(open(ck_restart_name, 'r'))
        start_epoch = d['start_epoch']
        max_valid_value = d['max_valid_value']
        minloss_value = d['minloss_value']

        net.load_state_dict(torch.load(ck_name, 'cpu'))
        optim.load_state_dict(torch.load(ck_optim_name, 'cpu'))

    # 用来快速跳过循环，快速检查代码问题
    use_one = False
    _do_one = False

    # 过程id

    for e in range(start_epoch, 100):
        optim_adjust.step(e)
        net.train()
        train_cla_acc = 0
        train_loss = 0

        train_gen = train_dataset.get_train_batch_gen(batch_count, batch_size, use_enhance=True, window_hw=net_in_hw)
        train_gen = preload_generator(train_gen)

        for b in range(batch_count):
            _t1 = time.time()
            ori_batch_im, ori_batch_label, ori_batch_ignore_mask, ori_batch_info = next(train_gen)
            _t2 = time.time()
            ori_batch_label_det, ori_batch_label_cla = np.split(ori_batch_label, [1,], 3)

            batch_im = torch.tensor(ori_batch_im, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255
            batch_label_cla = torch.tensor(ori_batch_label_cla, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            batch_label_mask = (batch_label_cla.max(1, keepdim=True)[0] > 0).type(torch.float32)

            batch_pred_cla = net(batch_im)
            _t3 = time.time()

            # 训练时检查像素准确率，这里的准确率按照类别平均检查
            with torch.no_grad():
                batch_label_cla_bool = (batch_label_cla > 0.5).type(torch.float32)
                batch_pred_cla_bool = (batch_pred_cla.clamp(0., 1.) > 0.5).type(torch.float32)
                cla_acc = (batch_pred_cla_bool * batch_label_cla_bool).sum() / (batch_label_mask.sum() + 1e-8)

                cla_acc = cla_acc.item()

                if time.time() - _last_auto_show_update_time > 60:
                    _last_auto_show_update_time = time.time()

                    vtb.images(batch_im, nrow=nrow, win='ori_ims', opts={'title': 'ori_ims'})
                    show_ims(vtb, batch_pred_cla.clamp(0, 1).mul(255).type(torch.uint8), 'pred_cla')
                    show_ims(vtb, batch_label_mask.mul(255).type(torch.uint8), 'ignore_mask')
                    show_ims(vtb, batch_label_cla.mul(255).type(torch.uint8), 'label_cla')

            _t4 = time.time()

            # loss = (3 * torch.abs(batch_pred_cla - batch_label_cla) * batch_label_mask).pow(2).sum(dim=(1, 2, 3)).__div__(batch_label_mask.sum(dim=(1, 2, 3)) + 1e-8).mean()
            loss = (3 * torch.abs(batch_pred_cla - batch_label_cla) * batch_label_mask).pow(2).sum(dim=(2, 3)).__div__(batch_label_mask.sum(dim=(2, 3)) + 1e-8).mean()
            # loss = (torch.pow(batch_pred_cla - batch_label_cla, 2)).sum(dim=1).mean()

            train_cla_acc += cla_acc
            train_loss += loss.item()

            print('epoch: {} count: {} train cla acc: {:.3f} loss: {:.3f}'.format(e, b, cla_acc, loss.item()))

            _t5 = time.time()

            optim.zero_grad()
            assert not np.isnan(loss.item()), 'Found loss Nan!'
            loss.backward()
            optim.step()

            _t6 = time.time()

            # print(f"{_t2-_t1:.2f}, ")

        train_cla_acc = train_cla_acc / batch_count
        train_loss = train_loss / batch_count

        sw.add_scalar('train_cla_acc', train_cla_acc, global_step=e)
        sw.add_scalar('train_loss', train_loss, global_step=e)

        # here to check eval
        if (e+1) % 1 == 0:
            with torch.no_grad():
                net.eval()

                gen = eval_dataset.get_eval_batch_gen(batch_size*2, window_hw=net_in_hw)
                # 使用预载函数，加速载入
                gen = preload_generator(gen)

                cla_score_table = {'epoch': e}
                for dt in match_distance_thresh_list:
                    cla_score_table[dt] = {
                        'found_pred': 0,        # 所有的假阳性
                        'fakefound_pred': 0,    # 所有的假阴性
                        'found_label': 0,       # 所有找到的标签
                        'nofound_label': 0,     # 所有找到的预测
                        'label_repeat': 0,      # 对应了多个pred的标签
                        'pred_repeat': 0,       # 对应了多个label的预测
                        'f1': None,
                        'recall': None,
                        'prec': None,
                    }

                    for cls_id in cla_class_ids:
                        cla_score_table[dt][cls_id] = {
                            'found_pred': 0,        # 所有的假阳性
                            'fakefound_pred': 0,    # 所有的假阴性
                            'found_label': 0,       # 所有找到的标签
                            'nofound_label': 0,     # 所有找到的预测
                            'label_repeat': 0,      # 对应了多个pred的标签
                            'pred_repeat': 0,       # 对应了多个label的预测
                            'f1': None,
                            'recall': None,
                            'prec': None,
                        }

                patch_count = 0
                eval_loss = 0.

                # 分割相关
                for ori_batch_im, ori_batch_label, ori_batch_ignore_mask, ori_batch_info in gen:

                    patch_count += 1

                    ori_batch_label_det, ori_batch_label_cla = np.split(ori_batch_label, [1, ], 3)

                    batch_im = torch.tensor(ori_batch_im, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255
                    batch_label_cla = torch.tensor(ori_batch_label_cla, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                    batch_label_mask = (batch_label_cla.max(1, keepdim=True)[0] > 0).type(torch.float32)

                    batch_pred_cla = net(batch_im)

                    loss = torch.log(torch.pow((batch_pred_cla - batch_label_cla) * batch_label_mask, 2).sum(dim=1) + 1).mean()

                    eval_loss += loss.item()

                    batch_pred_cla_final = batch_pred_cla * batch_label_mask

                    batch_label_mask = batch_label_mask.cpu().permute(0, 2, 3, 1).numpy()
                    batch_pred_cla_final = batch_pred_cla_final.cpu().permute(0, 2, 3, 1).numpy()

                    # 对每一个样本继续处理
                    # pi = 0
                    for each_label_cla, each_label_mask, each_pred_cla_final in\
                            zip(ori_batch_label_cla, batch_label_mask, batch_pred_cla_final):

                        cla_info = eval_utils.calc_a_sample_info_points_each_class(each_pred_cla_final, each_label_cla,
                                                                                   cls_list=cla_class_ids,
                                                                                   match_distance_thresh_list=match_distance_thresh_list,
                                                                                   use_post_pro=False)

                        if cla_info is not None:
                            for dt in match_distance_thresh_list:
                                for cls in cla_class_ids:
                                    cla_score_table[dt][cls]['found_pred'] +=       cla_info[cls][dt]['found_pred']
                                    cla_score_table[dt][cls]['fakefound_pred'] +=   cla_info[cls][dt]['fakefound_pred']
                                    cla_score_table[dt][cls]['found_label'] +=      cla_info[cls][dt]['found_label']
                                    cla_score_table[dt][cls]['nofound_label'] +=    cla_info[cls][dt]['nofound_label']
                                    cla_score_table[dt][cls]['pred_repeat'] +=      cla_info[cls][dt]['pred_repeat']
                                    cla_score_table[dt][cls]['label_repeat'] +=     cla_info[cls][dt]['label_repeat']

                                    cla_score_table[dt]['found_pred'] +=        cla_info[cls][dt]['found_pred']
                                    cla_score_table[dt]['fakefound_pred'] +=    cla_info[cls][dt]['fakefound_pred']
                                    cla_score_table[dt]['found_label'] +=       cla_info[cls][dt]['found_label']
                                    cla_score_table[dt]['nofound_label'] +=     cla_info[cls][dt]['nofound_label']
                                    cla_score_table[dt]['pred_repeat'] +=       cla_info[cls][dt]['pred_repeat']
                                    cla_score_table[dt]['label_repeat'] +=      cla_info[cls][dt]['label_repeat']

                # 计算cla的F1，精确率，召回率
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

                eval_loss = eval_loss / patch_count

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
                out_line = yaml.safe_dump(cla_score_table)

                print('epoch {}'.format(e), out_line)

                current_valid_value = cla_score_table[match_distance_thresh_list[0]]['f1']
                current_eval_loss = eval_loss

                # 保存最好的
                if current_valid_value > max_valid_value:
                    max_valid_value = current_valid_value
                    torch.save(net.state_dict(), ck_best_name)
                    yaml.safe_dump(cla_score_table, open(score_cla_best_name, 'w'))

                # 保存最小Loss的
                if current_eval_loss < minloss_value:
                    minloss_value = current_eval_loss
                    torch.save(net.state_dict(), ck_minloss_name)
                    yaml.safe_dump(cla_score_table, open(score_cla_minloss_name, 'w'))

                torch.save(net.state_dict(), ck_name)
                torch.save(optim.state_dict(), ck_optim_name)
                d = {'start_epoch': e + 1, 'max_valid_value': max_valid_value,
                     'minloss_value': minloss_value}
                yaml.safe_dump(d, open(ck_restart_name, 'w'))
                yaml.safe_dump(cla_score_table, open(score_cla_name, 'w'))

    sw.close()


if __name__ == '__main__':
    #from main_net9 import MainNet
    from main_net9 import MainNet2 as MainNet
    main(MainNet)

