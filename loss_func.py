import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union
# from lovasz_softmax import lovasz_hinge, lovasz_softmax


def a_loss(batch_label_pm, batch_pred_pm, pred_class_weight=None):
    # loss = (1 * (batch_label_pm - batch_pred_pm).abs()).pow(2)
    A = 1
    loss = ((1 + batch_label_pm * A) * (batch_label_pm - batch_pred_pm).abs()).pow(2)

    # weight = torch.where((batch_label_pm > 0.5).max(1, keepdim=True)[0], torch.full_like(batch_label_pm, 3),
    #                      torch.full_like(batch_label_pm, 1))
    loss = loss #* weight
    # pred_class_weight = torch.reshape(pred_class_weight, [1, -1, 1, 1])
    # loss = loss * pred_class_weight
    loss = loss.mean(dim=[2, 3]).sum(dim=1).mean()
    return loss


def a_cla_loss(batch_pred_det, batch_pred_cla: torch.Tensor, batch_label_cla: torch.Tensor, cla_weights: Union[torch.Tensor, str]='auto', threshold=0.5):
    '''
    注意，这里要求 batch_label_cla 是 onehot 向量，而不是类别标量
    :param batch_pred_det:
    :param batch_pred_cla:
    :param batch_label_cla:
    :param cla_weights:
    :param threshold:
    :return:
    '''
    assert batch_pred_cla.shape[1] == 1
    indicator = (batch_pred_det.detach() >= threshold).type(torch.float32)

    # cla_weights = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    # if isinstance(cla_weights, str) and cla_weights == 'auto':
    #     # 方法1，只使用权重1
    #     # cla_weights = torch.tensor([1]*batch_label_cla.shape[1], dtype=torch.float32, device=batch_label_cla.device)
    #     # cla_weights = cla_weights.reshape(1, -1, 1, 1)
    #     # 方法2，类别平衡
    #     batch_label_cla_masked = (batch_label_cla > 0.5).type(torch.float32) * indicator
    #     each_label_cla_pix_num = batch_label_cla_masked.sum(dim=(2, 3), keepdim=True)
    #     each_label_max_cls_pix_num = each_label_cla_pix_num.max(dim=1, keepdim=True)[0]
    #     cla_weights = torch.where(each_label_cla_pix_num > 0, each_label_max_cls_pix_num / each_label_cla_pix_num, torch.ones_like(each_label_cla_pix_num))
    #     # 防止log(0)出现负无穷
    #
    #     # # 新增，尝试
    #     # cla_weights = torch.log(cla_weights + 1) / 0.113328685307003    # 换底 0.113328685307003 = np.log(1.12)

    # batch_pred_cla = torch.where(batch_pred_cla > 1e-8, batch_pred_cla, batch_pred_cla + 1e-8)
    A = 1
    loss = ((1 + batch_label_cla * A) * (batch_label_cla - batch_pred_cla).abs()).pow(2)
    # loss = torch.pow(batch_pred_cla - batch_label_cla, 2)
    loss = (loss * indicator).sum() / indicator.sum()
    # loss = -torch.sum(cla_weights * indicator * torch.log(batch_pred_cla) * batch_label_cla, dim=(1, 2, 3)) /\
    #        (indicator.sum(dim=(1, 2, 3), dtype=torch.float32) + 1)
    return loss


def std_ce_loss(batch_label_pm, batch_pred_pm, pred_class_weight=None):
    '''
    batch_label_pm 接受热图，这里会自动将热图转换为类别id
    :param batch_label_pm:
    :param batch_pred_pm:
    :param pred_class_weight:
    :return:
    '''
    # label 值一定大于等于0.5，否则为0，以此为界限，将其转换成softmax可用label
    # 转换l1 label为 ce label
    pad_dim1 = torch.full([batch_label_pm.shape[0], 1, batch_label_pm.shape[2], batch_label_pm.shape[3]], 0.2,
                          device=batch_label_pm.device, dtype=batch_label_pm.dtype)
    batch_label_pm = torch.cat([pad_dim1, batch_label_pm], 1)
    batch_label_pm[:, 0].fill_(0.2)
    batch_label_pm = torch.argmax(batch_label_pm, 1)

    loss = F.cross_entropy(batch_pred_pm, batch_label_pm, weight=pred_class_weight)
    return loss


# def lovasz_softmax_loss(batch_label_pm, batch_pred_pm, pred_class_weight):
#     # label 值一定大于等于0.5，否则为0，以此为界限，将其转换成softmax可用label
#     batch_label_pm = batch_label_pm.clone()
#     batch_label_pm[:, 0].fill_(0.1)
#     batch_label_pm = torch.argmax(batch_label_pm, 1)
#     loss = lovasz_softmax(batch_pred_pm, batch_label_pm)
#     return loss
#
#
# def lovasz_hinge_loss(batch_label_pm, batch_pred_pm, pred_class_weight):
#     loss = 0
#     for c in range(batch_label_pm.shape[1]):
#         loss += lovasz_hinge(batch_pred_pm[:, c], batch_label_pm[:, c])
#     return loss


# def c_loss(batch_label_pm, batch_pred_pm, pred_class_weight=None):
#     # for det
#     batch_pred_pm = batch_pred_pm.sigmoid()
#     if pred_class_weight is None:
#         pred_class_weight = torch.ones(batch_pred_pm.shape[1], dtype=torch.float32, device=batch_label_pm.device)
#     pred_class_weight = pred_class_weight[None, :, None, None]
#     loss = -torch.mean(pred_class_weight * batch_label_pm * torch.log(batch_pred_pm) + (1-batch_label_pm) * torch.log(1-batch_pred_pm))
#     return loss
#
# def e_loss(batch_label_pm, batch_pred_pm, pred_class_weight=None):
#     loss = 3*(batch_pred_pm - batch_label_pm).abs()
#     # 当前设计一种变化的权重，使其可以动态变化
#     is_pos_bm = batch_label_pm > 0.5
#     # 利用log的变化曲线来动态调整权重
#     # weighted_loss = torch.log(1+loss) / np.log(1.06)
#     weighted_loss = torch.log(1+loss) / np.log(1.3)
#     loss = torch.where(is_pos_bm, weighted_loss, loss)
#     loss = loss.mean()
#     return loss


eps = 1e-3


def det_loss(batch_pred_det: torch.Tensor, batch_label_det: torch.Tensor, det_weights: Union[torch.Tensor, str]='auto'):
    '''
    注意，这里要求 y_true 是 onehot向量，而不是类别标量
    :param batch_pred_det:
    :param batch_label_cla:
    :param det_weights:
    :return:
    '''
    if isinstance(det_weights, str) and det_weights == 'auto':
        pos_n = batch_label_det[:, 1].sum()
        neg_n = batch_label_det[:, 0].sum()
        a = neg_n / pos_n
        det_weights = torch.tensor([1, a], dtype=torch.float32, device=batch_label_det.device)
        det_weights = torch.reshape(det_weights, [1, -1, 1, 1])
    elif isinstance(det_weights, torch.Tensor):
        det_weights = torch.reshape(det_weights, [1, -1, 1, 1])
    else:
        raise AssertionError('Unknow det_weights {}'.format(str(det_weights)))
    # 防止log(0)出现负无穷
    batch_pred_det = torch.where(batch_pred_det > 1e-8, batch_pred_det, batch_pred_det + 1e-8)
    loss = -torch.mean(det_weights * torch.log(batch_pred_det) * batch_label_det, dim=1, keepdim=True)
    return loss.mean()


def cla_loss(batch_pred_det, batch_pred_cla: torch.Tensor, batch_label_cla: torch.Tensor, cla_weights: Union[torch.Tensor, str]='auto', threshold=0.5):
    '''
    注意，这里要求 batch_label_cla 是 onehot 向量，而不是类别标量
    :param batch_pred_det:
    :param batch_pred_cla:
    :param batch_label_cla:
    :param cla_weights:
    :param threshold:
    :return:
    '''
    indicator = (batch_pred_det.detach()[:, 1:2] >= threshold).type(torch.float32)

    if isinstance(cla_weights, str) and cla_weights == 'auto':
        # 方法1，只使用权重1
        # cla_weights = torch.tensor([1]*batch_label_cla.shape[1], dtype=torch.float32, device=batch_label_cla.device)
        # cla_weights = cla_weights.reshape(1, -1, 1, 1)
        # 方法2，类别平衡
        batch_label_cla_masked = (batch_label_cla > 0.5).type(torch.float32) * indicator
        each_label_cla_pix_num = batch_label_cla_masked.sum(dim=(2, 3), keepdim=True)
        each_label_max_cls_pix_num = each_label_cla_pix_num.max(dim=1, keepdim=True)[0]
        cla_weights = torch.where(each_label_cla_pix_num > 0, each_label_max_cls_pix_num / each_label_cla_pix_num, torch.ones_like(each_label_cla_pix_num))
        # 防止log(0)出现负无穷

        # 新增，尝试
        cla_weights = torch.log(cla_weights + 1) / 0.113328685307003    # 换底 0.113328685307003 = np.log(1.12)

    batch_pred_cla = torch.where(batch_pred_cla > 1e-8, batch_pred_cla, batch_pred_cla + 1e-8)
    loss = -torch.sum(cla_weights * indicator * torch.log(batch_pred_cla) * batch_label_cla, dim=(1, 2, 3)) /\
           (indicator.sum(dim=(1, 2, 3), dtype=torch.float32) + 1)
    return loss.mean()


def cla_loss_v2(batch_pred_det, batch_pred_cla: torch.Tensor, batch_label_cla: torch.Tensor, batch_label_mask: torch.Tensor,
                cla_weights: Union[torch.Tensor, str]='auto', threshold=0.5):
    '''
    注意，这里要求 batch_label_cla 是 onehot 向量，而不是类别标量
    加入屏蔽功能，特定区域不计算Loss
    :param batch_pred_det:
    :param batch_pred_cla:
    :param batch_label_cla:
    :param cla_weights:
    :param threshold:
    :return:
    '''
    indicator = (batch_pred_det.detach()[:, 1:2] * batch_label_mask >= threshold).type(torch.float32)

    if isinstance(cla_weights, str) and cla_weights == 'auto':
        # 方法1，只使用权重1
        # cla_weights = torch.tensor([1]*batch_label_cla.shape[1], dtype=torch.float32, device=batch_label_cla.device)
        # cla_weights = cla_weights.reshape(1, -1, 1, 1)
        # 方法2，类别平衡
        batch_label_cla_masked = (batch_label_cla > 0.5).type(torch.float32) * indicator
        each_label_cla_pix_num = batch_label_cla_masked.sum(dim=(2, 3), keepdim=True)
        each_label_max_cls_pix_num = each_label_cla_pix_num.max(dim=1, keepdim=True)[0]
        cla_weights = torch.where(each_label_cla_pix_num > 0, each_label_max_cls_pix_num / each_label_cla_pix_num, torch.ones_like(each_label_cla_pix_num))
        # 防止log(0)出现负无穷

        # 新增，尝试
        cla_weights = torch.log(cla_weights + 1) / 0.113328685307003    # 换底 0.113328685307003 = np.log(1.12)

    batch_pred_cla = torch.where(batch_pred_cla > 1e-8, batch_pred_cla, batch_pred_cla + 1e-8)
    loss = -torch.sum(batch_label_mask * cla_weights * indicator * torch.log(batch_pred_cla) * batch_label_cla, dim=(1, 2, 3)) /\
           (indicator.sum(dim=(1, 2, 3), dtype=torch.float32) + 1)
    return loss.mean()


def joint_loss(batch_pred_det: torch.Tensor, batch_pred_cla: torch.Tensor, batch_label_cla: torch.Tensor, net_weights=0, sp_a=0, sp_b=0):
    batch_label_det = torch.zeros([batch_label_cla.shape[0], 2, batch_label_cla.shape[2], batch_label_cla.shape[3]],
                                  dtype=batch_label_cla.dtype, device=batch_label_cla.device)
    batch_label_det[:, 0] = batch_label_cla[:, 0]
    batch_label_det[:, 1] = 1 - batch_label_cla[:, 0]
    det = det_loss(batch_pred_det, batch_label_det)
    cla = cla_loss(batch_pred_det, batch_pred_cla, batch_label_cla)
    loss = det + cla * 0.1
    return loss


def joint_loss_v2(batch_pred_det: torch.Tensor, batch_pred_cla: torch.Tensor, batch_label_det: torch.Tensor,
                  batch_label_cla: torch.Tensor, batch_label_mask: torch.Tensor, net_weights=0, sp_a=0, sp_b=0):
    det = det_loss(batch_pred_det, batch_label_det)
    cla = cla_loss_v2(batch_pred_det, batch_pred_cla, batch_label_cla, batch_label_mask)
    loss = det + cla * 0.1
    return loss


def a_cla_loss_type3(batch_pred_det, batch_pred_cla: torch.Tensor, batch_label_cla: torch.Tensor, cla_weights: Union[torch.Tensor, str]='auto', threshold=0.5):
    '''
    注意，这里要求 batch_label_cla 是 onehot 向量，而不是类别标量
    :param batch_pred_det:
    :param batch_pred_cla:
    :param batch_label_cla:
    :param cla_weights:
    :param threshold:
    :return:
    '''
    indicator = (batch_pred_det.detach() >= threshold).type(torch.float32)

    cla_weights = torch.ones(1, batch_pred_cla.shape[1], 1, 1, dtype=torch.float32, device=batch_pred_det.device)
    if isinstance(cla_weights, str) and cla_weights == 'auto' and batch_pred_cla.shape[1] != 1:
        # 方法1，只使用权重1
        # cla_weights = torch.tensor([1]*batch_label_cla.shape[1], dtype=torch.float32, device=batch_label_cla.device)
        # cla_weights = cla_weights.reshape(1, -1, 1, 1)
        # 方法2，类别平衡
        batch_label_cla_masked = batch_label_cla * indicator
        each_label_cla_pix_num = batch_label_cla_masked.sum(dim=(0, 2, 3), keepdim=True)
        most_cla = torch.max(each_label_cla_pix_num, 1, keepdim=True)[0]
        each_label_cla_pix_num = torch.clamp_min(each_label_cla_pix_num, 1)
        cla_weights = most_cla / each_label_cla_pix_num
        cla_weights = torch.log(cla_weights + 1) / 0.113328685307003

    A = 1
    loss = ((1 + batch_label_cla * A) * cla_weights * (batch_label_cla - batch_pred_cla).abs()).pow(2)
    loss = (loss * indicator).sum() / indicator.sum()

    return loss

#
# if __name__ == '__main__':
#     batch_label_pm = torch.rand(3, 1, 16, 16)
#     batch_pred_pm = torch.rand(3, 2, 16, 16)
#     batch_pred_pm_l1 = torch.rand(3, 1, 16, 16)
#     pred_class_weight = torch.tensor([1, 2], dtype=torch.float32)
#
#     # loss = lovasz_softmax_loss(batch_label_pm, batch_pred_pm, pred_class_weight)
#     # print(loss)
#
#     loss = std_ce_loss(batch_label_pm, batch_pred_pm, pred_class_weight)
#     print(loss)
#
#     batch_label_pm = torch.rand(3, 6, 16, 16)
#     batch_pred_pm = torch.rand(3, 6, 16, 16)
#     pred_class_weight = torch.tensor([1, 2, 1, 2, 1, 1], dtype=torch.float32)
#
#     loss = a_loss(batch_label_pm, batch_pred_pm, pred_class_weight)
#     print(loss)
#
#     batch_label_pm = torch.rand(3, 6, 16, 16)
#     batch_pred_pm = torch.rand(3, 6, 16, 16)
#     loss = lovasz_hinge_loss(batch_label_pm, batch_pred_pm, pred_class_weight)
#     print(loss)
