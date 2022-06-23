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
    dataset_path, process_control, net_save_dir, match_distance_thresh_list, b1_is_ce_loss, b2_is_ce_loss, b3_is_ce_loss, make_cla_is_det,\
    new_img_test_in_dir, new_img_test_out_dir
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
from skimage.draw import disk as sk_disk


use_heatmap_nms = True
use_single_pair = True
pred_heatmap_thresh_list = (0.3,)

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


def make_mix_pic(im, pred, label, prob=0.4):
    color_label = (label > 0.5).astype(np.uint8) * 255
    color_pred = (pred > prob).astype(np.uint8) * 255
    pad = np.zeros([*color_pred.shape[:2], 1], dtype=color_pred.dtype)
    color_hm = np.concatenate([color_pred, color_label, pad], axis=-1)
    mix_pic = np.where(np.any(color_hm > 0, -1, keepdims=True), color_hm, im)
    return mix_pic


def draw_cla_im(im, pts, cls):
    cls_color = {
        0: (255, 0, 0),
        1: (255, 128, 0),
        2: (0, 128, 255),
        3: (0, 255, 0),
        4: (128, 255, 0),
    }
    assert len(pts) == len(cls)
    im = im.copy()
    for pt, c in zip(pts, cls):
        cv2.circle(im, tuple(pt)[::-1], 3, cls_color[c], 1)
    return im


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

    os.makedirs(new_img_test_in_dir, exist_ok=True)
    os.makedirs(new_img_test_out_dir, exist_ok=True)

    # 分割相关
    for heatmap_thresh in pred_heatmap_thresh_list:
        for im_name in os.listdir(new_img_test_in_dir):
            print(f'process {im_name}')
            in_im_path = f'{new_img_test_in_dir}/{im_name}'
            out_im_path = f'{new_img_test_out_dir}/{im_name}'

            im = imageio.imread(in_im_path)
            im = cv2.resize(im, (im.shape[0]//2, im.shape[1]//2), interpolation=cv2.INTER_AREA)
            im = im[:, :, :3]

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
            # pred_det_rough_pts = eval_utils.get_pts_from_hm(pred_det_rough_pm, heatmap_thresh)

            # det fine
            if use_heatmap_nms:
                pred_det_final_pm[..., 0] = pred_det_final_pm[..., 0] * heatmap_nms(pred_det_final_pm[..., 0])

            pred_det_post_pts = eval_utils.get_pts_from_hm(pred_det_final_pm, heatmap_thresh)

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

            draw_im = draw_cla_im(im, pred_det_post_pts, pred_cla_pts_cla)

            imageio.imwrite(out_im_path, draw_im)


if __name__ == '__main__':
    from main_net11 import MainNet
    main(MainNet)
