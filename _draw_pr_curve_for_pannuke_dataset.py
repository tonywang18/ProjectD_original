import yaml
import matplotlib.pyplot as plt
import os
import numpy as np
from a_config import project_root


target_dir = os.path.join(project_root, 'pn_pred_circle_train_out_dir')
# target_dir = os.path.join(project_root, 'pn_pred_circle_valid_out_dir')
# target_dir = os.path.join(project_root, 'pn_pred_circle_test_out_dir')
dist_th = 6

recall_list = []
prec_list = []
hm_th_list = []


for dir_name in os.listdir(target_dir):
    dir_path = os.path.join(target_dir, dir_name)
    if not os.path.isdir(dir_path):
        print('Will skip dir', dir_path)
        continue
    try:
        float(dir_name)
    except ValueError:
        print('Will skip dir', dir_path)
        continue

    a2_det_yml = os.path.join(dir_path, 'all_det_a2.txt')
    data = yaml.safe_load(open(a2_det_yml, 'r'))
    recall_list.append(data[dist_th]['recall'])
    prec_list.append(data[dist_th]['prec'])
    hm_th_list.append(float(dir_name))

recall_list = np.array(recall_list)
prec_list = np.array(prec_list)
hm_th_list = np.array(hm_th_list)


def draw_pr_curve(recall_list, prec_list, out_im):

    pr_sorted_ids = np.argsort(recall_list)

    pr_recall_list = recall_list[pr_sorted_ids]
    pr_prec_list = prec_list[pr_sorted_ids]

    plt.figure(1, clear=True)
    plt.plot(pr_recall_list, pr_prec_list, linewidth=2, markersize=12, label="PR-Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.savefig(out_im)


def draw_pr2_curve(recall_list, prec_list, hm_th_list, out_im):

    pr2_sorted_ids = np.argsort(hm_th_list)

    pr2_hm_th_list = hm_th_list[pr2_sorted_ids]
    pr2_recall_list = recall_list[pr2_sorted_ids]
    pr2_prec_list = prec_list[pr2_sorted_ids]

    plt.figure(1, clear=True)
    plt.plot(pr2_hm_th_list, pr2_recall_list, linewidth=2, markersize=12, label="Thresh-Recall-Curve")
    plt.plot(pr2_hm_th_list, pr2_prec_list, linewidth=2, markersize=12, label="Thresh-Precision-Curve")
    plt.xlabel('Thresh')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(out_im)


draw_pr_curve(recall_list, prec_list, os.path.join(target_dir, 'pr-curve.png'))
draw_pr2_curve(recall_list, prec_list, hm_th_list, os.path.join(target_dir, 'pr2-curve.png'))
