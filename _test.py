import os
import torch
import torch.nn.functional as F
import time
import numpy as np
import yaml
from dataset_reader import DatasetReader
from tensorboardX import SummaryWriter
import cv2
import yaml
from my_py_lib.auto_show_running import AutoShowRunning

from a_config import project_root, device, net_in_hw, net_out_hw, batch_size, epoch, batch_count, is_train_from_recent_checkpoint, dataset_path
# from eval_utils import fusion_im_contours, class_map_to_contours, calc_a_sample_info
from my_py_lib.preload_generator import preload_generator
from my_py_lib.numpy_tool import one_hot
from eval_utils import calc_a_sample_info


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

save_dir = os.path.join(project_root, 'save')
log_dir = os.path.join(project_root, 'logs')

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


def main(NetClass, mode='det'):
    assert mode == 'det'
    torch.set_grad_enabled(False)
    _last_auto_show_update_time = 0

    model_id = NetClass.model_id

    # asr_show_num_hw = [3, 4]
    # asr = AutoShowRunning(out_hw=[750, 900], show_num_hw=asr_show_num_hw)

    ck_name         = '{}/{}_model.pt'          .format(save_dir, model_id)
    ck_best_name    = '{}/{}_model_best.pt'     .format(save_dir, model_id)
    ck_optim_name   = '{}/{}_optim.pt'          .format(save_dir, model_id)
    ck_restart_name = '{}/{}_restart.yml'       .format(save_dir, model_id)
    ck_extra_name   = '{}/{}_extra.txt'         .format(save_dir, model_id)
    score_name      = '{}/{}_score.txt'         .format(save_dir, model_id)
    score_best_name = '{}/{}_score_best.txt'    .format(save_dir, model_id)

    logdir = '{}_{}' .format(log_dir, model_id)
    sw = SummaryWriter(logdir)

    eval_dataset = DatasetReader(dataset_path, mode, is_train=True)

    net = NetClass()
    net = net.to(device)
    net.eval()

    if os.path.isfile(ck_name):
        net.load_state_dict(torch.load(ck_best_name))

    gen = eval_dataset.get_eval_batch_gen(batch_size*4, window_hw=net_in_hw)
    # 使用预载函数，加速载入
    # gen = preload_generator(gen)

    score_table = {'fake_found': 0}                     # 找错了的
    for cls in [0]:
        score_table[cls] = {
                'found_with_true_cls': 0,               # 找到了，包含iou是最大和不是最大的情况
                'found_with_error_cls': 0,              # 找到了，但却分类错误
                'found_with_true_with_max_iou': 0,      # 找到了，最高iou的轮廓分类正确
                'found_with_true_without_max_iou': 0,   # 找到了，不是最高iou的轮廓分类正确
                'nofound': 0,                           # 没找到
                }

    # patch_count = 0

    # 分割相关
    for batch_im, batch_label in gen:
        # patch_count += 1

        batch_im = torch.tensor(batch_im, dtype=torch.float32) / 255
        batch_im = batch_im.permute(0, 3, 1, 2)
        batch_im = batch_im.to(device)

        _, _, batch_pred = net(batch_im)
        batch_pred = batch_pred.permute(0, 2, 3, 1).cpu().numpy()

        # 对每一个样本继续处理
        for each_label_cm, each_pred_cm in zip(batch_label, batch_pred):
            info = calc_a_sample_info(each_pred_cm, each_label_cm, cls_list=[0], cls_score_thresh=0.7, match_iou_thresh=0.001, use_post_pro=False)
            score_table['fake_found'] += info['fake_found']
            for cls in [0]:
                score_table[cls]['found_with_true_cls'] += info[cls]['found_with_true_cls']
                score_table[cls]['found_with_error_cls'] += info[cls]['found_with_error_cls']
                score_table[cls]['found_with_true_with_max_iou'] += info[cls]['found_with_true_with_max_iou']
                score_table[cls]['found_with_true_without_max_iou'] += info[cls]['found_with_true_without_max_iou']
                score_table[cls]['nofound'] += info[cls]['nofound']

    all_fake_found = score_table['fake_found']
    all_nofound = 0
    all_found_with_true_cls = 0
    all_found_with_error_cls = 0

    for cls in [0]:
        all_nofound += score_table[cls]['nofound']
        all_found_with_true_cls += score_table[cls]['found_with_true_cls']
        all_found_with_error_cls += score_table[cls]['found_with_error_cls']

    # 打印评分
    out_line = 'all_fake_found {:5d} all_nofound {:5d} all_found_with_true_cls {:5d} all_found_with_error_cls {:5d}\n'.format(
        all_fake_found, all_nofound, all_found_with_true_cls, all_found_with_error_cls)

    for cls in [0]:
        out_line += 'cls {} found_with_true_cls {:5d} found_with_error_cls {:5d} found_with_true_with_max_iou {:5d}' \
                    ' found_with_true_without_max_iou {:5d} nofound {:5d}\n'.\
            format(cls, score_table[cls]['found_with_true_cls'], score_table[cls]['found_with_error_cls'],
                   score_table[cls]['found_with_true_with_max_iou'], score_table[cls]['found_with_true_without_max_iou'],
                   score_table[cls]['nofound'])

    print(out_line)

    # current_valid_value = all_found_with_true_cls * 10 - all_found_with_error_cls - all_nofound * 15 - all_fake_found * 5

    # open(ck_extra_name, 'w').write(out_line)
    # yaml.safe_dump(score_table, open(score_best_name, 'w'))
    # yaml.safe_dump(score_table, open(score_name, 'w'))


if __name__ == '__main__':
    from main_net1 import MainNet
    main(MainNet)

