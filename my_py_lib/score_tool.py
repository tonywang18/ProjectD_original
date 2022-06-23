'''
算分工具，用于求解常见的 精确率，召回率，Fn分数
'''


def calc_score_fn(prec: float, recall: float, n: float, *, eps=1e-8):
    '''
    计算fn分数
    :param prec:    精确率
    :param recall:  召回率
    :param n:       指标n。例如为1，代表计算f1分数，若为2代表计算f2分数，若为0.5代表计算f0.5分数。
                    大于1时为偏重召回率，小于1时为偏重精确率
    :return: 返回Fn分数
    '''
    fn = (1 + n**2) * prec * recall / max(prec * n**2 + recall, eps)
    return fn


def calc_score_prec_recall(n_label_found: int, n_label_nofound: int, n_pred_found: int, n_pred_fakefound: int):
    '''
    计算分数 prec, recall
    :param n_label_found:       发现的标签数量
    :param n_label_nofound:     没有发现的标签的数量
    :param n_pred_found:        发现的预测的数量
    :param n_pred_fakefound:    虚假的预测的数量
    :return:
    '''
    assert n_label_found >= 0 and n_label_nofound >= 0 and n_pred_found >= 0 and n_pred_fakefound >= 0, 'Error! No allow any params is lower than 0.'
    # 使用max是为了避免影响结果和避免除零错误
    prec = n_pred_found / max(n_pred_found + n_pred_fakefound, 1)
    recall = n_label_found / max(n_label_found + n_label_nofound, 1)
    return prec, recall


def calc_score_f05_f1_f2_prec_recall(n_label_found: int, n_label_nofound: int, n_pred_found: int, n_pred_fakefound: int):
    '''
    聚合计算分数 f0.5, f1, f2, prec, recall
    用于简单使用
    :param n_label_found:       发现的标签数量
    :param n_label_nofound:     没有发现的标签的数量
    :param n_pred_found:        发现的预测的数量
    :param n_pred_fakefound:    虚假的预测的数量
    :return:
    '''
    assert n_label_found >= 0 and n_label_nofound >= 0 and n_pred_found >= 0 and n_pred_fakefound >= 0, 'Error! No allow any params is lower than 0.'
    prec, recall = calc_score_prec_recall(n_label_found, n_label_nofound, n_pred_found, n_pred_fakefound)
    f05 = calc_score_fn(prec, recall, 0.5)
    f1 = calc_score_fn(prec, recall, 1)
    f2 = calc_score_fn(prec, recall, 2)
    return f05, f1, f2, prec, recall
