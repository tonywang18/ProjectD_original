'''
包围框评分工具
'''

import numpy as np
from .score_tool import calc_score_f05_f1_f2_prec_recall
from .bbox_tool import calc_bbox_iou_NtoM, check_bboxes, get_bboxes_shortest_link_pair
from .list_tool import list_multi_get_with_bool, list_multi_get_with_ids


def calc_bbox_score(pred_bboxes, pred_classes, label_bboxes, label_classes, classes_list,
                    match_iou_thresh_list=(0.3, 0.5, 0.7),
                    use_single_pair=False, *, return_lookup_table=True):
    '''
    通用包围框评估
    将会返回一个分数字典
    当预测与标签的IOU大于指定阈值时，将会认定为真阳性
    结构为
    类别-IOU分数-X
    X:
    found_pred      真阳性，预测正确的数量
    fakefound_pred  假阳性，预测失败的数量
    found_label     真阳性，标签被正确匹配到的数量
    nofound_label   假阴性，没有任何成功匹配的标签数量
    pred_repeat     当use_single_pair为False时，一个预测可以同时匹配多个标签，该度量将会统计匹配数量大于1的预测的数量
    label_repeat    当use_single_pair为False时，一个标签可以同时匹配多个预测，该度量将会统计匹配数量大于1的标签的数量
    f05             f0.5分数
    f1              f1分数
    f2              f2分数
    prec            准确率
    recall          召回率

    :param pred_bboxes:             预测的包围框
    :param pred_classes:            预测的类别
    :param label_bboxes:            标签的包围框
    :param label_classes:           标签的类别
    :param classes_list:            要评估的类别列表
    :param match_iou_thresh_list:   多个评估阈值
    :param use_single_pair:         若为真，则使用一个预测只匹配一个标签。如果假，每个预测都可以匹配多个标签
    :param return_lookup_table:     若为真，将返回一个查找表
    :return:
    '''
    classes_list = np.array(classes_list).tolist()
    match_iou_thresh_list = np.array(match_iou_thresh_list).tolist()

    score_table = {}

    if return_lookup_table:
        lookup_table = {}

    if len(pred_bboxes) == 0 or len(label_bboxes) == 0:
        for cls in classes_list:
            score_table[cls] = {}

            if return_lookup_table:
                lookup_table[cls] = {}

            for iou_th in match_iou_thresh_list:
                score_table[cls][iou_th] = {}
                score_table[cls][iou_th]['found_pred'] = 0
                score_table[cls][iou_th]['fakefound_pred'] = len(pred_bboxes)
                score_table[cls][iou_th]['found_label'] = 0
                score_table[cls][iou_th]['nofound_label'] = len(label_bboxes)
                score_table[cls][iou_th]['pred_repeat'] = 0
                score_table[cls][iou_th]['label_repeat'] = 0
                score_table[cls][iou_th]['f05'] = 0.
                score_table[cls][iou_th]['f1'] = 0.
                score_table[cls][iou_th]['f2'] = 0.
                score_table[cls][iou_th]['prec'] = 0.
                score_table[cls][iou_th]['recall'] = 0.

                if return_lookup_table:
                    lookup_table[iou_th] = {}
                    score_table[cls][iou_th]['found_pred'] = []
                    score_table[cls][iou_th]['fakefound_pred'] = list(pred_bboxes)
                    score_table[cls][iou_th]['found_label'] = []
                    score_table[cls][iou_th]['nofound_label'] = list(label_bboxes)
                    score_table[cls][iou_th]['pred_repeat'] = []
                    score_table[cls][iou_th]['label_repeat'] = []

        if return_lookup_table:
            return score_table, lookup_table
        else:
            return score_table

    pred_bboxes = np.float32(pred_bboxes)
    label_bboxes = np.float32(label_bboxes)
    check_bboxes(pred_bboxes)
    check_bboxes(label_bboxes)

    pred_classes = np.int32(pred_classes)
    label_classes = np.int32(label_classes)

    assert pred_classes.ndim == 1
    assert label_classes.ndim == 1
    assert len(pred_bboxes) == len(pred_classes)
    assert len(label_bboxes) == len(label_classes)

    for cls in classes_list:
        score_table[cls] = {}

        # 筛选出当前类的轮廓
        pred_selected_bools = np.array(pred_classes, np.int32) == cls
        label_selected_bools = np.array(label_classes, np.int32) == cls
        selected_pred_bboxes = pred_bboxes[pred_selected_bools]
        selected_label_bboxes = label_bboxes[label_selected_bools]

        if return_lookup_table:
            lookup_table[cls] = {}

        ious_table = calc_bbox_iou_NtoM(selected_pred_bboxes, selected_label_bboxes)

        # 计算不同IOU下的f0.5,f1,f2,recall,prec分数
        for iou_th in match_iou_thresh_list:
            score_table[cls][iou_th] = {}

            label_found_count = np.zeros(len(selected_label_bboxes), np.int32)
            pred_found_count = np.zeros(len(selected_pred_bboxes), np.int32)

            if len(selected_label_bboxes) > 0:
                if not use_single_pair:
                    for pi, pred_contour in enumerate(selected_pred_bboxes):
                        ious = ious_table[pi]
                        close_bools = ious >= iou_th
                        label_found_count[close_bools] += 1
                        pred_found_count[pi] += np.array(close_bools, np.int32).sum()
                else:
                    pred_ids, label_ids, _ = get_bboxes_shortest_link_pair(selected_pred_bboxes, selected_label_bboxes, iou_th, pair_ious=ious_table)
                    for i in pred_ids:
                        pred_found_count[i] += 1
                    for i in label_ids:
                        label_found_count[i] += 1

            ids_found_pred = np.argwhere(pred_found_count > 0).squeeze(1)
            ids_fakefound_pred = np.argwhere(pred_found_count == 0).squeeze(1)
            ids_found_label = np.argwhere(label_found_count > 0).squeeze(1)
            ids_nofound_label = np.argwhere(label_found_count == 0).squeeze(1)
            ids_pred_repeat = np.argwhere(pred_found_count > 1).squeeze(1)
            ids_label_repeat = np.argwhere(label_found_count > 1).squeeze(1)

            found_pred = len(ids_found_pred)
            fakefound_pred = len(ids_fakefound_pred)

            found_label = len(ids_found_label)
            nofound_label = len(ids_nofound_label)

            pred_repeat = len(ids_pred_repeat)
            label_repeat = len(ids_label_repeat)

            f05, f1, f2, prec, recall = calc_score_f05_f1_f2_prec_recall(found_label, nofound_label, found_pred, fakefound_pred)

            score_table[cls][iou_th]['found_pred'] = int(found_pred)
            score_table[cls][iou_th]['fakefound_pred'] = int(fakefound_pred)
            score_table[cls][iou_th]['found_label'] = int(found_label)
            score_table[cls][iou_th]['nofound_label'] = int(nofound_label)
            score_table[cls][iou_th]['pred_repeat'] = int(pred_repeat)
            score_table[cls][iou_th]['label_repeat'] = int(label_repeat)
            score_table[cls][iou_th]['f05'] = float(f05)
            score_table[cls][iou_th]['f1'] = float(f1)
            score_table[cls][iou_th]['f2'] = float(f2)
            score_table[cls][iou_th]['prec'] = float(prec)
            score_table[cls][iou_th]['recall'] = float(recall)

            if return_lookup_table:
                lookup_table[cls][iou_th] = {}
                lookup_table[cls][iou_th]['found_pred'] = list_multi_get_with_ids(selected_pred_bboxes, ids_found_pred)
                lookup_table[cls][iou_th]['fakefound_pred'] = list_multi_get_with_ids(selected_pred_bboxes, ids_fakefound_pred)
                lookup_table[cls][iou_th]['found_label'] = list_multi_get_with_ids(selected_label_bboxes, ids_found_label)
                lookup_table[cls][iou_th]['nofound_label'] = list_multi_get_with_ids(selected_label_bboxes, ids_nofound_label)
                lookup_table[cls][iou_th]['pred_repeat'] = list_multi_get_with_ids(selected_pred_bboxes, ids_pred_repeat)
                lookup_table[cls][iou_th]['label_repeat'] = list_multi_get_with_ids(selected_label_bboxes, ids_label_repeat)

    if return_lookup_table:
        return score_table, lookup_table
    else:
        return score_table


def summary_bbox_score(scores, cls_list, match_iou_thresh_list):
    '''
    对多个分数表进行合算，得到统计分数表
    其中 found_pred, fakefound_pred, found_label, nofound_label, pred_repeat, label_repeat 将会被累加
    其中 f1, prec, recall 将会被求平均
    :param scores:                      多个分数表
    :param cls_list:                    要检查的分类
    :param match_iou_thresh_list:       多个匹配IOU
    :return:
    '''
    score_table = {}
    for cls in cls_list:
        score_table[cls] = {}
        for iou_th in match_iou_thresh_list:
            score_table[cls][iou_th] = {}
            score_table[cls][iou_th]['found_pred'] = 0
            score_table[cls][iou_th]['fakefound_pred'] = 0
            score_table[cls][iou_th]['found_label'] = 0
            score_table[cls][iou_th]['nofound_label'] = 0
            score_table[cls][iou_th]['pred_repeat'] = 0
            score_table[cls][iou_th]['label_repeat'] = 0
            score_table[cls][iou_th]['f05'] = 0.
            score_table[cls][iou_th]['f1'] = 0.
            score_table[cls][iou_th]['f2'] = 0.
            score_table[cls][iou_th]['prec'] = 0.
            score_table[cls][iou_th]['recall'] = 0.

    for score in scores:
        for cls in cls_list:
            for iou_th in match_iou_thresh_list:
                score_table[cls][iou_th]['found_pred'] += score[cls][iou_th]['found_pred']
                score_table[cls][iou_th]['fakefound_pred'] += score[cls][iou_th]['fakefound_pred']
                score_table[cls][iou_th]['found_label'] += score[cls][iou_th]['found_label']
                score_table[cls][iou_th]['nofound_label'] += score[cls][iou_th]['nofound_label']
                score_table[cls][iou_th]['pred_repeat'] += score[cls][iou_th]['pred_repeat']
                score_table[cls][iou_th]['label_repeat'] += score[cls][iou_th]['label_repeat']
                score_table[cls][iou_th]['f05'] += score[cls][iou_th]['f05'] / len(scores)
                score_table[cls][iou_th]['f1'] += score[cls][iou_th]['f1'] / len(scores)
                score_table[cls][iou_th]['f2'] += score[cls][iou_th]['f2'] / len(scores)
                score_table[cls][iou_th]['prec'] += score[cls][iou_th]['prec'] / len(scores)
                score_table[cls][iou_th]['recall'] += score[cls][iou_th]['recall'] / len(scores)

    return score_table
