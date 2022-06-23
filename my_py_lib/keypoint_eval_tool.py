'''
关键点评分工具
'''

import numpy as np
from .score_tool import calc_score_f05_f1_f2_prec_recall
from .point_tool import get_shortest_link_pair


def calc_keypoint_score(pred_centers, pred_cls, label_centers, label_cls, cls_list, match_distance_thresh_list=(5, 7, 9, 11), use_single_pair=False):
    '''
    通用关键点评估
    将会返回一个分数字典
    当预测与标签距离小于评估距离时，将会认定为真阳性
    结构为
    类别-评估距离-X
    X:
    found_pred      真阳性，预测正确的数量
    fakefound_pred  假阳性，预测失败的数量
    found_label     真阳性，标签被正确匹配到的数量
    nofound_label   假阴性，没有任何成功匹配的标签数量
    pred_repeat     当use_single_pair为False时，一个预测可以同时匹配多个标签，该度量将会统计匹配数量大于1的预测的数量
    label_repeat    当use_single_pair为False时，一个标签可以同时匹配多个预测，该度量将会统计匹配数量大于1的标签的数量

    :param pred_centers:    预测的中心点
    :param pred_cls:        预测的类别
    :param label_centers:   标签的中心点
    :param label_cls:       标签的类别
    :param cls_list:        要评估的类别列表
    :param match_distance_thresh_list: 多个评估阈值
    :param use_single_pair: 若为真，则使用一个预测点只匹配一个标签。如果假，每个预测点都可以匹配多个标签
    :return:
    '''
    score_table = {}

    if len(label_centers) == 0 or len(pred_centers) == 0:
        for cls in cls_list:
            score_table[cls] = {}
            for dt in match_distance_thresh_list:
                score_table[cls][dt] = {}
                score_table[cls][dt]['found_pred'] = 0
                score_table[cls][dt]['fakefound_pred'] = len(pred_centers)
                score_table[cls][dt]['found_label'] = 0
                score_table[cls][dt]['nofound_label'] = len(label_centers)
                score_table[cls][dt]['pred_repeat'] = 0
                score_table[cls][dt]['label_repeat'] = 0
                score_table[cls][dt]['f05'] = 0.
                score_table[cls][dt]['f1'] = 0.
                score_table[cls][dt]['f2'] = 0.
                score_table[cls][dt]['prec'] = 0.
                score_table[cls][dt]['recall'] = 0.
        return score_table

    pred_centers = np.asarray(pred_centers, np.float32)
    if len(pred_centers) == 0:
        pred_centers = np.reshape(pred_centers, [-1, 2])
    pred_cls = np.asarray(pred_cls, np.int32)
    label_centers = np.reshape(np.asarray(label_centers, np.float32), [-1, 2])
    if len(label_centers) == 0:
        label_centers = np.reshape(label_centers, [-1, 2])
    label_cls = np.asarray(label_cls, np.int32)

    assert pred_centers.ndim == 2 and pred_centers.shape[1] == 2
    assert label_centers.ndim == 2 and label_centers.shape[1] == 2
    assert pred_cls.ndim == 1
    assert label_cls.ndim == 1
    assert len(pred_centers) == len(pred_cls)
    assert len(label_centers) == len(label_cls)

    for cls in cls_list:
        score_table[cls] = {}
        pred_selected_bools = np.array(pred_cls, np.int32) == cls
        label_selected_bools = np.array(label_cls, np.int32) == cls
        selected_pred_centers = pred_centers[pred_selected_bools]
        selected_label_centers = label_centers[label_selected_bools]

        for dt in match_distance_thresh_list:
            score_table[cls][dt] = {}

            label_found_count = np.zeros(len(selected_label_centers), np.int32)
            pred_found_count = np.zeros(len(selected_pred_centers), np.int32)

            if not use_single_pair:
                for pi, pred_center in enumerate(selected_pred_centers):
                    if len(selected_label_centers) != 0:
                        dists = np.linalg.norm(pred_center[None,] - selected_label_centers, axis=1)
                        close_bools = dists <= dt
                        label_found_count[close_bools] += 1
                        pred_found_count[pi] += np.array(close_bools, np.int32).sum()
            else:
                pred_pt_ids, label_pt_ids, _ = get_shortest_link_pair(selected_pred_centers, selected_label_centers, dt)
                for i in pred_pt_ids:
                    pred_found_count[i] += 1
                for i in label_pt_ids:
                    label_found_count[i] += 1

            found_pred = (pred_found_count > 0).sum()
            fakefound_pred = (pred_found_count == 0).sum()

            found_label = (label_found_count > 0).sum()
            nofound_label = (label_found_count == 0).sum()

            pred_repeat = (pred_found_count > 1).sum()
            label_repeat = (label_found_count > 1).sum()

            f05, f1, f2, prec, recall = calc_score_f05_f1_f2_prec_recall(found_label, nofound_label, found_pred, fakefound_pred)

            score_table[cls][dt]['found_pred'] = int(found_pred)
            score_table[cls][dt]['fakefound_pred'] = int(fakefound_pred)
            score_table[cls][dt]['found_label'] = int(found_label)
            score_table[cls][dt]['nofound_label'] = int(nofound_label)
            score_table[cls][dt]['pred_repeat'] = int(pred_repeat)
            score_table[cls][dt]['label_repeat'] = int(label_repeat)
            score_table[cls][dt]['f05'] = float(f05)
            score_table[cls][dt]['f1'] = float(f1)
            score_table[cls][dt]['f2'] = float(f2)
            score_table[cls][dt]['prec'] = float(prec)
            score_table[cls][dt]['recall'] = float(recall)

    return score_table


def summary_keypoint_score(scores, cls_list, match_distance_thresh_list):
    '''
    对多个分数表进行合算，得到统计分数表
    其中 found_pred, fakefound_pred, found_label, nofound_label, pred_repeat, label_repeat 将会被累加
    其中 f1, prec, recall 将会被求平均
    :param scores:                      多个分数表
    :param cls_list:                    要检查的分类
    :param match_distance_thresh_list:  多个匹配距离
    :return:
    '''
    score_table = {}
    for cls in cls_list:
        score_table[cls] = {}
        for dt in match_distance_thresh_list:
            score_table[cls][dt] = {}
            score_table[cls][dt]['found_pred'] = 0
            score_table[cls][dt]['fakefound_pred'] = 0
            score_table[cls][dt]['found_label'] = 0
            score_table[cls][dt]['nofound_label'] = 0
            score_table[cls][dt]['pred_repeat'] = 0
            score_table[cls][dt]['label_repeat'] = 0
            score_table[cls][dt]['f05'] = 0.
            score_table[cls][dt]['f1'] = 0.
            score_table[cls][dt]['f2'] = 0.
            score_table[cls][dt]['prec'] = 0.
            score_table[cls][dt]['recall'] = 0.

    for score in scores:
        for cls in cls_list:
            for dt in match_distance_thresh_list:
                score_table[cls][dt]['found_pred'] += score[cls][dt]['found_pred']
                score_table[cls][dt]['fakefound_pred'] += score[cls][dt]['fakefound_pred']
                score_table[cls][dt]['found_label'] += score[cls][dt]['found_label']
                score_table[cls][dt]['nofound_label'] += score[cls][dt]['nofound_label']
                score_table[cls][dt]['pred_repeat'] += score[cls][dt]['pred_repeat']
                score_table[cls][dt]['label_repeat'] += score[cls][dt]['label_repeat']
                score_table[cls][dt]['f05'] += score[cls][dt]['f05'] / len(scores)
                score_table[cls][dt]['f1'] += score[cls][dt]['f1'] / len(scores)
                score_table[cls][dt]['f2'] += score[cls][dt]['f2'] / len(scores)
                score_table[cls][dt]['prec'] += score[cls][dt]['prec'] / len(scores)
                score_table[cls][dt]['recall'] += score[cls][dt]['recall'] / len(scores)

    return score_table
