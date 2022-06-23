'''
轮廓评分工具
'''

import numpy as np
from .score_tool import calc_score_f05_f1_f2_prec_recall
from .contour_tool import calc_iou_with_contours_1toN, calc_dice_contours, calc_iou_with_contours_NtoM, get_contours_shortest_link_pair
from .list_tool import list_multi_get_with_bool, list_multi_get_with_ids
import prettytable


def calc_contour_score(pred_contours, pred_classes, label_contours, label_classes, classes_list,
                       match_iou_thresh_list=(0.3, 0.5, 0.7),
                       use_single_pair=False, *, return_lookup_table=True):
    '''
    通用轮廓评估
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
    f05             f0.5分数
    f1              f1分数
    f2              f2分数
    prec            准确率
    recall          召回率
    pred_area       预测轮廓的面积
    label_area      标签轮廓的面积
    overlap_area    预测轮廓与标签轮廓的重叠面积
    dice            dice分数

    :param pred_contours:           预测的包围框
    :param pred_classes:            预测的类别
    :param label_contours:          标签的包围框
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

    if len(pred_contours) == 0 or len(label_contours) == 0:
        for cls in classes_list:
            score_table[cls] = {}
            score_table[cls]['pred_area'] = 0.
            score_table[cls]['label_area'] = 0.
            score_table[cls]['overlap_area'] = 0.
            score_table[cls]['dice'] = 0.

            if return_lookup_table:
                lookup_table[cls] = {}

            for iou_th in match_iou_thresh_list:
                score_table[cls][iou_th] = {}
                score_table[cls][iou_th]['found_pred'] = 0
                score_table[cls][iou_th]['fakefound_pred'] = len(pred_contours)
                score_table[cls][iou_th]['found_label'] = 0
                score_table[cls][iou_th]['nofound_label'] = len(label_contours)
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
                    score_table[cls][iou_th]['fakefound_pred'] = list(pred_contours)
                    score_table[cls][iou_th]['found_label'] = []
                    score_table[cls][iou_th]['nofound_label'] = list(label_contours)
                    score_table[cls][iou_th]['pred_repeat'] = []
                    score_table[cls][iou_th]['label_repeat'] = []

        if return_lookup_table:
            return score_table, lookup_table
        else:
            return score_table

    pred_contours = [np.float32(c) for c in pred_contours]
    label_contours = [np.float32(c) for c in label_contours]

    pred_classes = np.int32(pred_classes)
    label_classes = np.int32(label_classes)

    assert pred_classes.ndim == 1
    assert label_classes.ndim == 1
    assert len(pred_contours) == len(pred_classes)
    assert len(label_contours) == len(label_classes)

    for cls in classes_list:
        score_table[cls] = {}
        
        # 筛选出当前类的轮廓
        pred_selected_bools = np.array(pred_classes, np.int32) == cls
        label_selected_bools = np.array(label_classes, np.int32) == cls
        selected_pred_contours = list_multi_get_with_bool(pred_contours, pred_selected_bools)
        selected_label_contours = list_multi_get_with_bool(label_contours, label_selected_bools)

        # 计算dice分数
        dice, overlap_area, pred_area, label_area = calc_dice_contours(selected_pred_contours, selected_label_contours, return_area=True)
        score_table[cls]['pred_area'] = pred_area
        score_table[cls]['label_area'] = label_area
        score_table[cls]['overlap_area'] = overlap_area
        score_table[cls]['dice'] = dice

        if return_lookup_table:
            lookup_table[cls] = {}

        ious_table = calc_iou_with_contours_NtoM(selected_pred_contours, selected_label_contours)

        # 计算不同IOU下的f0.5,f1,f2,recall,prec分数
        for iou_th in match_iou_thresh_list:
            score_table[cls][iou_th] = {}

            label_found_count = np.zeros(len(selected_label_contours), np.int32)
            pred_found_count = np.zeros(len(selected_pred_contours), np.int32)

            if len(selected_label_contours) > 0:
                if not use_single_pair:
                    for pi, pred_contour in enumerate(selected_pred_contours):
                        ious = ious_table[pi]
                        close_bools = ious >= iou_th
                        label_found_count[close_bools] += 1
                        pred_found_count[pi] += np.array(close_bools, np.int32).sum()
                else:
                    pred_ids, label_ids, _ = get_contours_shortest_link_pair(selected_pred_contours, selected_label_contours, iou_th, pair_ious=ious_table)
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
                lookup_table[cls][iou_th]['found_pred'] = list_multi_get_with_ids(selected_pred_contours, ids_found_pred)
                lookup_table[cls][iou_th]['fakefound_pred'] = list_multi_get_with_ids(selected_pred_contours, ids_fakefound_pred)
                lookup_table[cls][iou_th]['found_label'] = list_multi_get_with_ids(selected_label_contours, ids_found_label)
                lookup_table[cls][iou_th]['nofound_label'] = list_multi_get_with_ids(selected_label_contours, ids_nofound_label)
                lookup_table[cls][iou_th]['pred_repeat'] = list_multi_get_with_ids(selected_pred_contours, ids_pred_repeat)
                lookup_table[cls][iou_th]['label_repeat'] = list_multi_get_with_ids(selected_label_contours, ids_label_repeat)

    if return_lookup_table:
        return score_table, lookup_table
    else:
        return score_table


def summary_contour_score(scores, classes_list, match_iou_thresh_list):
    '''
    对多个分数表进行合算，得到统计分数表
    其中 found_pred, fakefound_pred, found_label, nofound_label, pred_repeat, label_repeat 将会被累加
    其中 f1, prec, recall 将会被求平均
    其中 pred_area, label_area, overlap_area 将会被累加
    其中 dice 将会被求平均
    :param scores:                      多个分数表
    :param classes_list:                要检查的分类
    :param match_iou_thresh_list:       多个匹配IOU
    :return:
    '''
    classes_list = np.array(classes_list).tolist()
    match_iou_thresh_list = np.array(match_iou_thresh_list).tolist()

    score_table = {}
    for cls in classes_list:
        score_table[cls] = {}
        score_table[cls]['pred_area'] = 0.
        score_table[cls]['label_area'] = 0.
        score_table[cls]['overlap_area'] = 0.
        score_table[cls]['dice'] = 0.
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
        for cls in classes_list:
            score_table[cls]['pred_area'] += score[cls]['pred_area']
            score_table[cls]['label_area'] += score[cls]['label_area']
            score_table[cls]['overlap_area'] += score[cls]['overlap_area']
            score_table[cls]['dice'] += score[cls]['dice'] / len(scores)
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


def make_show_text(score_table, class_ids, match_iou_thresh_list):
    '''
    把分数表打印为易读文本

    示例如下
    | class |  dice |    pred_area |   label_area | overlap_area |
    |     1 | 0.720 | 21261576.000 | 19290693.000 | 14601321.499 |
    |     2 | 0.503 | 14234351.500 | 12560113.000 |  6732315.954 |
    |     5 | 0.727 |   379825.000 |   284914.500 |   241507.125 |
    |     6 | 0.573 |   586936.000 |   719623.000 |   374471.442 |
    |     - |     - |            - |            - |            - |
    | class | iou_th | found_pred | fakefound_pred | found_label | nofound_label | pred_repeat | label_repeat |   f05 |    f1 |    f2 |  prec | recall |
    |     1 |  0.300 |        115 |             25 |         115 |            74 |           0 |            0 | 0.768 | 0.699 | 0.642 | 0.821 |  0.608 |
    |     1 |  0.500 |         92 |             48 |          92 |            97 |           0 |            0 | 0.614 | 0.559 | 0.513 | 0.657 |  0.487 |
    |     1 |  0.800 |          8 |            132 |           8 |           181 |           0 |            0 | 0.053 | 0.049 | 0.045 | 0.057 |  0.042 |
    |     - |      - |          - |              - |           - |             - |           - |            - |     - |     - |     - |     - |      - |
    |     2 |  0.300 |        145 |            138 |         145 |            48 |           0 |            0 | 0.547 | 0.609 | 0.687 | 0.512 |  0.751 |
    |     2 |  0.500 |        119 |            164 |         119 |            74 |           0 |            0 | 0.449 | 0.500 | 0.564 | 0.420 |  0.617 |
    |     2 |  0.800 |         20 |            263 |          20 |           173 |           0 |            0 | 0.075 | 0.084 | 0.095 | 0.071 |  0.104 |
    |     - |      - |          - |              - |           - |             - |           - |            - |     - |     - |     - |     - |      - |
    |     5 |  0.300 |         15 |              4 |          15 |             3 |           0 |            0 | 0.798 | 0.811 | 0.824 | 0.789 |  0.833 |
    |     5 |  0.500 |         12 |              7 |          12 |             6 |           0 |            0 | 0.638 | 0.649 | 0.659 | 0.632 |  0.667 |
    |     5 |  0.800 |          0 |             19 |           0 |            18 |           0 |            0 | 0.000 | 0.000 | 0.000 | 0.000 |  0.000 |
    |     - |      - |          - |              - |           - |             - |           - |            - |     - |     - |     - |     - |      - |
    |     6 |  0.300 |         83 |             21 |          83 |            79 |           0 |            0 | 0.718 | 0.624 | 0.552 | 0.798 |  0.512 |
    |     6 |  0.500 |         72 |             32 |          72 |            90 |           0 |            0 | 0.623 | 0.541 | 0.479 | 0.692 |  0.444 |
    |     6 |  0.800 |         12 |             92 |          12 |           150 |           0 |            0 | 0.104 | 0.090 | 0.080 | 0.115 |  0.074 |
    |     - |      - |          - |              - |           - |             - |           - |            - |     - |     - |     - |     - |      - |

    :param score_table:
    :param class_ids:
    :param match_iou_thresh_list:
    :return:
    '''
    out_text = ''

    ## 打印Dice相关
    out_table = prettytable.PrettyTable(['class', 'dice', 'pred_area', 'label_area', 'overlap_area'])
    out_table.set_style(prettytable.MSWORD_FRIENDLY)
    out_table.float_format = ".3"
    out_table.align = 'r'
    for cls in class_ids:
        out_table.add_row(
            [cls,
             score_table[cls]['dice'],
             score_table[cls]['pred_area'],
             score_table[cls]['label_area'],
             score_table[cls]['overlap_area'],
             ]
        )
    out_table.add_row(['-'] * 5)
    out_text += out_table.get_string() + '\n'

    ## 打印目标检测相关
    out_table = prettytable.PrettyTable(['class', 'iou_th', 'found_pred', 'fakefound_pred', 'found_label', 'nofound_label',
                                         'pred_repeat', 'label_repeat', 'f05', 'f1', 'f2', 'prec', 'recall'])
    out_table.set_style(prettytable.MSWORD_FRIENDLY)
    out_table.float_format = ".3"
    out_table.align = 'r'

    for cls in class_ids:
        for iou_th in match_iou_thresh_list:
            out_table.add_row(
                [cls, iou_th,
                 score_table[cls][iou_th]['found_pred'],
                 score_table[cls][iou_th]['fakefound_pred'],
                 score_table[cls][iou_th]['found_label'],
                 score_table[cls][iou_th]['nofound_label'],
                 score_table[cls][iou_th]['pred_repeat'],
                 score_table[cls][iou_th]['label_repeat'],
                 score_table[cls][iou_th]['f05'],
                 score_table[cls][iou_th]['f1'],
                 score_table[cls][iou_th]['f2'],
                 score_table[cls][iou_th]['prec'],
                 score_table[cls][iou_th]['recall'],
                 ]
            )
        out_table.add_row(['-'] * 13)
    out_text += out_table.get_string() + '\n'

    return out_text