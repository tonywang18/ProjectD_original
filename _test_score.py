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
from my_py_lib.auto_show_running import AutoShowRunning

from a_config import project_root, device, net_in_hw, net_out_hw, batch_size, epoch, batch_count, is_train_from_recent_checkpoint, dataset_path
# from eval_utils import fusion_im_contours, class_map_to_contours, calc_a_sample_info
from my_py_lib.preload_generator import preload_generator
from my_py_lib.numpy_tool import one_hot
from my_py_lib.image_over_scan_wrapper import ImageOverScanWrapper
from my_py_lib.coords_over_scan_gen import n_step_scan_coords_gen
from eval_utils import calc_a_sample_info, simple_nms_post_process, calc_metric


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

    pred_out_dir = 'pred_out_dir'
    os.makedirs(pred_out_dir, exist_ok=True)

    eval_dataset = DatasetReader(dataset_path, mode, is_train=True, pt_radius=6)

    net = NetClass()
    net = net.to(device)
    net.eval()

    eval_weight_path = ck_name
    if os.path.isfile(eval_weight_path):
        net.load_state_dict(torch.load(eval_weight_path))

    score_table = {
        'label_found_num': 0,
        'label_nofound_num': 0,
        'pred_fake_found_num': 0,
        'label_num': 0,
        'pred_num': 0,
        'recall': 0,
        'prec': 0,
        'f1': 0,
    }

    # 分割相关
    for pid in range(len(eval_dataset)):
        im, label, label_pos, label_cls = eval_dataset.get_item(pid, use_enhance=False, window_hw=None, return_pos=True)
        label = np.asarray(label, np.float32)
        pred = np.zeros_like(label)

        wim = ImageOverScanWrapper(im)
        wpred = ImageOverScanWrapper(pred)
        coords_gen = n_step_scan_coords_gen(im.shape[:2], net_in_hw, n_step=1)

        for yx_start, yx_end in coords_gen:
            patch_im = wim.get(yx_start, yx_end)
            patch_pred = wpred.get(yx_start, yx_end)

            patch_im = torch.tensor(patch_im, dtype=torch.float32) / 255
            patch_im = patch_im[None,].permute(0, 3, 1, 2)
            patch_im = patch_im.to(device)
            _, _, out_patch_pred = net(patch_im)
            # out_patch_pred = post_process(out_patch_pred)
            out_patch_pred = out_patch_pred.permute(0, 2, 3, 1).cpu().numpy()[0]
            patch_pred = np.maximum(patch_pred, out_patch_pred)
            wpred.set(yx_start, yx_end, patch_pred)

        d = calc_metric(wpred.im, label_pos)
        score_table['label_found_num'] += d['label_found_num']
        score_table['label_nofound_num'] += d['label_nofound_num']
        score_table['pred_fake_found_num'] += d['pred_fake_found_num']
        score_table['label_num'] += d['label_num']
        score_table['pred_num'] += d['pred_num']

        print(d)

    score_table['recall'] = score_table['label_found_num'] / (score_table['label_num'] + 1e-8)
    score_table['prec'] = (score_table['pred_num'] - score_table['pred_fake_found_num']) / (score_table['pred_num'] + 1e-8)
    score_table['f1'] = 2 * (score_table['prec'] * score_table['recall']) / (score_table['prec'] + score_table['recall'] + 1e-8)

    print(score_table)


if __name__ == '__main__':
    from main_net1 import MainNet
    main(MainNet)

