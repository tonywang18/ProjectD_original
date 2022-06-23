import yaml
import numpy as np


def calc_score(cla_true, cla_fake, cla_all):
    prec = cla_true / max(cla_true + cla_fake, 1e-8)
    recall = cla_true / max(cla_all, 1e-8)
    f1 = 2 * (prec * recall) / max(prec + recall, 1e-8)
    return prec, recall, f1


if __name__ == '__main__':
    pn_info = [r'D:\bio-totem\project\projectD\cla_pn_pred_circle_train_out_dir\0.4\all_cla.txt',
               r'D:\bio-totem\project\projectD\cla_pn_pred_circle_valid_out_dir\0.4\all_cla.txt',
               r'D:\bio-totem\project\projectD\cla_pn_pred_circle_test_out_dir\0.4\all_cla.txt',]

    '''
    0 -> 0
    4 -> 0
    1 -> 1
    2 -> 2
    3 -> 2
    '''
    # true fake all prec recall f1
    c0_d = [0, 0, 0, 0, 0, 0]
    c1_d = [0, 0, 0, 0, 0, 0]
    c2_d = [0, 0, 0, 0, 0, 0]

    for f in pn_info:
        info = yaml.safe_load(open(f, 'r'))
        info = info[6]

        c0_d[0] += info[0]['found_pred'] + info[4]['found_pred']
        c0_d[1] += info[0]['fakefound_pred'] + info[4]['fakefound_pred']
        c0_d[2] += info[0]['found_label'] + info[0]['nofound_label'] + info[4]['found_label'] + info[4]['nofound_label']

        c1_d[0] += info[1]['found_pred']
        c1_d[1] += info[1]['fakefound_pred']
        c1_d[2] += info[1]['found_label'] + info[1]['nofound_label']

        c2_d[0] += info[2]['found_pred'] + info[3]['found_pred']
        c2_d[1] += info[2]['fakefound_pred'] + info[3]['fakefound_pred']
        c2_d[2] += info[2]['found_label'] + info[2]['nofound_label'] + info[3]['found_label'] + info[3]['nofound_label']

    c0_d[3], c0_d[4], c0_d[5] = calc_score(c0_d[0], c0_d[1], c0_d[2])
    c1_d[3], c1_d[4], c1_d[5] = calc_score(c1_d[0], c1_d[1], c1_d[2])
    c2_d[3], c2_d[4], c2_d[5] = calc_score(c2_d[0], c2_d[1], c2_d[2])

    print('true fake all prec recall f1')
    print(c0_d)
    print(c1_d)
    print(c2_d)
