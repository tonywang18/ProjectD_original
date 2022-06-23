import torch
import numpy as np
from dataset_reader_PanNuke import DatasetReader
import cv2
import imageio

from a_config import project_root, device, net_in_hw, net_out_hw, batch_size, \
    process_control, net_save_dir, match_distance_thresh_list, b1_is_ce_loss, b2_is_ce_loss, b3_is_ce_loss
import eval_utils

from big_pic_result import BigPicPatch
from heatmap_nms import heatmap_nms


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


class DetInterface:
    def __init__(self, weight_type, device='cuda:0'):
        from main_net11 import MainNet
        NetClass = MainNet
        torch.set_grad_enabled(False)

        model_id = NetClass.model_id
        ck_name = '{}/{}_model.pt'.format(net_save_dir, model_id)
        ck_best_name = '{}/{}_model_best.pt'.format(net_save_dir, model_id)
        # ck_minloss_name = '{}/{}_model_minloss.pt'.format(net_save_dir, model_id)

        # 定义网络
        b1_out_dim = 1 + 1 if b1_is_ce_loss else 1
        b2_out_dim = 1 + 1 if b2_is_ce_loss else 1
        b3_out_dim = cla_class_num + 1 if b3_is_ce_loss else cla_class_num
        net = NetClass(3, b1_out_dim, b2_out_dim, b3_out_dim)

        net.enabled_b2_branch = True
        net.enabled_b3_branch = True

        start_epoch = 150

        if weight_type == 'last':
            print('Will load last weight.')
            new_ck_name = get_pg_name(ck_name, start_epoch, process_control)
            net.load_state_dict(torch.load(new_ck_name, 'cpu'))
            print('load model success')
        elif weight_type == 'best':
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

        self.net = net

    def __call__(self, im: np.ndarray, det_thresh=0.3):
        assert im.ndim == 3 and im.shape[-1] == 3
        # 分割相关
        heatmap_thresh = det_thresh

        # im = cv2.resize(im, (im.shape[0] // 2, im.shape[1] // 2), interpolation=cv2.INTER_AREA)
        # im = im[:, :, :3]

        # 运行区
        wim = BigPicPatch(1 + 1 + cla_class_num, [im], (0, 0), window_hw=net_in_hw, level_0_patch_hw=(1, 1),
                          custom_patch_merge_pipe=eval_utils.patch_merge_func, patch_border_pad_value=255, ignore_patch_near_border_ratio=0.5)
        gen = wim.batch_get_im_patch_gen(batch_size * 3)
        for batch_info, batch_patch in gen:
            batch_patch = torch.tensor(np.asarray(batch_patch), dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.
            batch_pred_det, batch_pred_det2, batch_pred_cla = self.net(batch_patch)

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

            wim.batch_update_result(batch_info, batch_pred)

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

        pred_cla_pts = []
        pred_cla_pts_cla_probs = []
        pred_cla_pts.extend(pred_det_post_pts)
        pred_cla_pts_cla_probs.extend(eval_utils.get_cls_pts_from_hm_2(pred_cla_pts, pred_cla_final_pm_hm_before))

        # pred_cla_pts_cla = []
        # pred_cla_pts_cla.extend(eval_utils.get_cls_pts_from_hm(pred_cla_pts, pred_cla_final_pm_hm_before))
        # draw_im = draw_cla_im(im, pred_det_post_pts, pred_cla_pts_cla)
        # cv2.imshow('asd', draw_im[..., ::-1])
        # cv2.waitKey()

        # 打包
        pred_cla_pts = np.array(pred_cla_pts, np.float32).reshape([-1, 2])
        pad = np.zeros([pred_cla_pts.shape[0], 1], np.float32)
        pred_cla_pts_cla_probs = np.array(pred_cla_pts_cla_probs, np.float32).reshape([-1, 5])

        arr = np.concatenate([pred_cla_pts, pad, pred_cla_pts_cla_probs], axis=1)

        return arr


if __name__ == '__main__':
    det = DetInterface('best')
    im = imageio.imread(r"D:\bio-totem\project\projectD\data_2\train\img3\img3.bmp")
    arr = det(im)
    print(arr.shape)
