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
from my_py_lib import im_tool
from my_py_lib.auto_show_running import AutoShowRunning

from a_config import project_root, device, net_in_hw, net_out_hw, batch_size, epoch, batch_count, eval_which_checkpoint, dataset_path, process_control, net_save_dir
# from eval_utils import fusion_im_contours, class_map_to_contours, calc_a_sample_info
from my_py_lib.preload_generator import preload_generator
from my_py_lib.numpy_tool import one_hot
from my_py_lib.image_over_scan_wrapper import ImageOverScanWrapper
from my_py_lib.coords_over_scan_gen import n_step_scan_coords_gen
from eval_utils import calc_a_sample_info
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


pred_det_train_out_dir = 'pred_det_train_out_dir'
pred_det_valid_out_dir = 'pred_det_valid_out_dir'
pred_cla_train_out_dir = 'pred_cla_train_out_dir'
pred_cla_valid_out_dir = 'pred_cla_valid_out_dir'


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


def patch_merge_func(self, patch_result, patch_result_new, patch_mask):
    '''
    默认的滑窗图合并函数，合并时取最大值
    :param self:                引用大图块自身，用于实现某些特殊用途，一般不使用
    :param patch_result:        当前滑窗区域的结果
    :param patch_result_new:    新的滑窗区域的结果
    :param patch_mask:          当前掩码，用于特殊用途，这里不使用
    :return: 返回合并后结果和更新的掩码
    '''
    new_result = patch_result + patch_result_new
    new_mask = patch_mask + 1
    return new_result, new_mask


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

    start_epoch = 250

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

        # 跳过det数据集
        if did < 2:
            continue

        os.makedirs(out_dir, exist_ok=True)

        if pg_id == 1 and did >= 2:
            break

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

            # pred_det_a1 = np.zeros([*label_det.shape[:2], 1], np.float32)
            # pred_det_a2 = np.zeros([*label_det.shape[:2], 1], np.float32)
            # pred_cla = np.zeros([*label_det.shape[:2], 4], np.float32)

            # wim = ImageOverScanWrapper(im)
            # wpred_det_a1 = ImageOverScanWrapper(pred_det_a1)
            # wpred_det_a2 = ImageOverScanWrapper(pred_det_a2)
            # wpred_cla = ImageOverScanWrapper(pred_cla)
            # coords_gen = n_step_scan_coords_gen(im.shape[:2], net_in_hw, n_step=2)

            #
            bpp_im = BigPicPatch(1+5, [im], (0, 0), window_hw=net_in_hw, level_0_patch_hw=(1, 1), custom_patch_merge_pipe=patch_merge_func)
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

            pred_det_bin = im_tool.bin_infill((pred_det >= 0.8).astype(np.uint8)).astype(np.float32)

            pred_det_a1 = pred_det
            pred_det_a2 = pred_det * (1 - pred_cla[:, :, 0:1])

            mix_pic_det_a1 = make_mix_pic(im, pred_det_a1, label_det, 0.8)
            mix_pic_det_a2 = make_mix_pic(im, pred_det_a2, label_det, 0.5)

            pred_cla_b1 = (pred_cla >= pred_cla.max(2, keepdims=True)).astype(pred_cla.dtype)
            pred_cla_b1 = pred_det_bin * pred_cla_b1
            pred_cla_b1 = pred_cla_b1[:, :, 1:]

            pred_cla_b2 = (np.clip(pred_cla[..., 1:] * pred_det, 0, 1) * 255).astype(np.uint8)

            mix_pic_cla_b1 = []
            if did >= 2:
                for i in range(4):
                    mix_pic_cla_b1.append(make_mix_pic(im, pred_cla_b1[..., i:i+1], label_cla[..., i:i+1], 0.4))

            imageio.imwrite(os.path.join(out_dir, '{}_det_a1.png'.format(im_basename)), mix_pic_det_a1)
            imageio.imwrite(os.path.join(out_dir, '{}_det_a2.png'.format(im_basename)), mix_pic_det_a2)
            if did >= 2:
                for i in range(4):
                    imageio.imwrite(os.path.join(out_dir, '{}_cla_b1_{}.png'.format(im_basename, i)), mix_pic_cla_b1[i])

            if did >= 2:
                for i in range(4):
                    imageio.imwrite(os.path.join(out_dir, '{}_cla_b2_{}.png'.format(im_basename, i)), pred_cla_b2[:, :, i])


if __name__ == '__main__':
    from main_net7 import MainNet
    main(MainNet)

