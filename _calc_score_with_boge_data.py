import pickle
import gzip
import numpy as np
import yaml
import cv2
import os
from my_py_lib import contour_tool
import imageio


pannuke_cla_name = ('Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial', 'Det')
boge_cla_name = ('epit', 'lymph', 'other')


def calc_score(cla_true, cla_fake, cla_all):
    prec = cla_true / max(cla_true + cla_fake, 1e-8)
    recall = cla_true / max(cla_all, 1e-8)
    f1 = 2 * (prec * recall) / max(prec + recall, 1e-8)
    return prec, recall, f1


def test_cla():
    fold_id = 1
    pannuke_data_path = f'pannuke_packed_dataset/fold_{fold_id}.pkl.gz'
    boge_cls_data_path = f'boge_data/fold_{fold_id}_result_all.pkl'
    result_path = f'boge_data/cla_result_{fold_id}.yml'
    each_result_path = f'boge_data/cla_each_result_{fold_id}.yml'

    is_draw_pic = True
    draw_result_dir = f'boge_data/cla_each_result_{fold_id}_pic'
    if is_draw_pic:
        os.makedirs(draw_result_dir, exist_ok=True)

    pannuke_data = pickle.loads(gzip.decompress(open(pannuke_data_path, 'rb').read()))
    im_types = pannuke_data['im_types']
    del pannuke_data

    boge_cls_data = pickle.load(open(boge_cls_data_path, 'rb'))

    # 筛选乳腺组织
    assert len(im_types) == len(boge_cls_data)
    boge_cls_data = [boge_cls_data[i] for i in range(len(im_types)) if im_types[i] == 'Breast']

    '''
    - item - show_img_arr
           - d0-5 - [N,yx_conts]conts
                  - [N, 3]probs
    '''
    # 图像总数
    n_item = len(boge_cls_data)

    # 每个图像的分数的列表
    each_item_score_list = []

    # pn_cls_set = set()

    # 开始计算每幅图像分数
    for item_id, item in enumerate(boge_cls_data):
        show_im, data = item

        if is_draw_pic:
            imageio.imwrite(f'{draw_result_dir}/{item_id}.jpg', show_im)

        # 每个图像的分数表
        item_score = {
            'id': item_id,
            'epti': {
                'true': 0,
                'fake': 0,
                'all': 0,
                'prec': 0,
                'recall': 0,
                'f1': 0,
            },
            'lymph': {
                'true': 0,
                'fake': 0,
                'all': 0,
                'prec': 0,
                'recall': 0,
                'f1': 0,
            },
            'other': {
                'true': 0,
                'fake': 0,
                'all': 0,
                'prec': 0,
                'recall': 0,
                'f1': 0,
            }
        }

        for pn_cls in data.keys():
            # pn_cls_set.add(pn_cls)

            boge_cont_list, boge_probs_list = data[pn_cls]

            for cont, probs in zip(boge_cont_list, boge_probs_list):

                if pn_cls in (0, 4):
                    # 0 Neoplastic -> 0 epti
                    # 4 Epithelial -> 0 epti
                    item_score['epti']['all'] += 1
                    if np.argmax(probs) == 0:
                        item_score['epti']['true'] += 1
                    elif np.argmax(probs) == 1:
                        item_score['lymph']['fake'] += 1
                    elif np.argmax(probs) == 2:
                        item_score['other']['fake'] += 1
                    else:
                        raise AssertionError('Error! Unknow class!')

                elif pn_cls in (1,):
                    # 1 Inflammatory -> 1 lymph
                    item_score['lymph']['all'] += 1
                    if np.argmax(probs) == 0:
                        item_score['epti']['fake'] += 1
                    elif np.argmax(probs) == 1:
                        item_score['lymph']['true'] += 1
                    elif np.argmax(probs) == 2:
                        item_score['other']['fake'] += 1
                    else:
                        raise AssertionError('Error! Unknow class!')

                elif pn_cls in (2, 3):
                    # 2 Connective -> 2 other
                    # 3 Dead -> 2 other
                    item_score['other']['all'] += 1
                    if np.argmax(probs) == 0:
                        item_score['epti']['fake'] += 1
                    elif np.argmax(probs) == 1:
                        item_score['lymph']['fake'] += 1
                    elif np.argmax(probs) == 2:
                        item_score['other']['true'] += 1
                    else:
                        raise AssertionError('Error! Unknow class!')

                elif pn_cls == 5:
                    # 这个是检测类，需要忽略
                    continue
                else:
                    # 未知的类别，一定是哪里出错了
                    raise AssertionError('Error! Unknow class id! Please check again!')

            item_score['epti']['prec'], item_score['epti']['recall'], item_score['epti']['f1'] = \
                calc_score(item_score['epti']['true'], item_score['epti']['fake'], item_score['epti']['all'])

            item_score['lymph']['prec'], item_score['lymph']['recall'], item_score['lymph']['f1'] = \
                calc_score(item_score['lymph']['true'], item_score['lymph']['fake'], item_score['lymph']['all'])

            item_score['other']['prec'], item_score['other']['recall'], item_score['other']['f1'] = \
                calc_score(item_score['other']['true'], item_score['other']['fake'], item_score['other']['all'])

        each_item_score_list.append(item_score)

    # 总分计算
    all_score = {
        'epti': {
            'true': 0,
            'fake': 0,
            'all': 0,
            'prec': 0,
            'recall': 0,
            'f1': 0,
        },
        'lymph': {
            'true': 0,
            'fake': 0,
            'all': 0,
            'prec': 0,
            'recall': 0,
            'f1': 0,
        },
        'other': {
            'true': 0,
            'fake': 0,
            'all': 0,
            'prec': 0,
            'recall': 0,
            'f1': 0,
        }
    }

    for item_score in each_item_score_list:
        all_score['epti']['true'] += item_score['epti']['true']
        all_score['epti']['fake'] += item_score['epti']['fake']
        all_score['epti']['all'] += item_score['epti']['all']
        
        all_score['lymph']['true'] += item_score['lymph']['true']
        all_score['lymph']['fake'] += item_score['lymph']['fake']
        all_score['lymph']['all'] += item_score['lymph']['all']
        
        all_score['other']['true'] += item_score['other']['true']
        all_score['other']['fake'] += item_score['other']['fake']
        all_score['other']['all'] += item_score['other']['all']

    all_score['epti']['prec'], all_score['epti']['recall'], all_score['epti']['f1'] = \
        calc_score(all_score['epti']['true'], all_score['epti']['fake'], all_score['epti']['all'])

    all_score['lymph']['prec'], all_score['lymph']['recall'], all_score['lymph']['f1'] = \
        calc_score(all_score['lymph']['true'], all_score['lymph']['fake'], all_score['lymph']['all'])

    all_score['other']['prec'], all_score['other']['recall'], all_score['other']['f1'] = \
        calc_score(all_score['other']['true'], all_score['other']['fake'], all_score['other']['all'])

    text = yaml.safe_dump(all_score)
    print(text)
    open(result_path, 'w').write(text)

    open(each_result_path, 'w').write(yaml.safe_dump(each_item_score_list))
    print('complete')


def test_seg():
    fold_id = 3
    pannuke_data_path = f'pannuke_packed_dataset/fold_{fold_id}.pkl.gz'
    boge_seg_data_path = f'boge_data/fold_{fold_id}_mask_pred.pkl'
    result_path = f'boge_data/seg_result_{fold_id}.yml'
    each_result_path = f'boge_data/seg_each_result_{fold_id}.yml'

    is_draw_pic = True
    draw_result_dir = f'boge_data/seg_each_result_{fold_id}_pic'
    if is_draw_pic:
        os.makedirs(draw_result_dir, exist_ok=True)

    pannuke_data = pickle.loads(gzip.decompress(open(pannuke_data_path, 'rb').read()))
    im_types = pannuke_data['im_types']
    lable_cls_conts_list = pannuke_data['labels']
    del pannuke_data

    boge_seg_data_list = pickle.load(open(boge_seg_data_path, 'rb'))

    # 筛选乳腺组织
    assert len(im_types) == len(boge_seg_data_list)
    boge_seg_data_list = [boge_seg_data_list[i] for i in range(len(im_types)) if im_types[i] == 'Breast']
    lable_cls_conts_list = [lable_cls_conts_list[i] for i in range(len(im_types)) if im_types[i] == 'Breast']

    '''
    - item - show_img_arr
           - d0-5 - [N,yx_conts]conts
                  - [N, 3]probs
    '''
    # 图像总数
    n_item = len(boge_seg_data_list)

    # 每个图像的分数的列表
    each_item_score_list = []

    # pn_cls_set = set()

    # 开始计算每幅图像分数
    for item_id, item in enumerate(boge_seg_data_list):
        cur_pred_conts = contour_tool.tr_cv_to_my_contours(item)
        cur_label_cls_conts = lable_cls_conts_list[item_id]
        cur_label_conts = []
        for k in cur_label_cls_conts.keys():
            if k == 5:
                continue
            cur_label_conts.extend(cur_label_cls_conts[k])

        merge_cur_pred_conts = contour_tool.merge_multi_contours(cur_pred_conts)
        merge_cur_label_conts = contour_tool.merge_multi_contours(cur_label_conts)

        inter_conts = []
        for pred_cont in merge_cur_pred_conts:
            conts = contour_tool.inter_contours_1toN(pred_cont, merge_cur_label_conts)
            inter_conts.extend(conts)

        merge_inter_conts = contour_tool.merge_multi_contours(inter_conts)

        pred_area = sum([contour_tool.calc_contour_area(c) for c in merge_cur_pred_conts])
        label_area = sum([contour_tool.calc_contour_area(c) for c in merge_cur_label_conts])
        inter_area = sum([contour_tool.calc_contour_area(c) for c in merge_inter_conts])

        dice = 2 * inter_area / max(label_area + pred_area, 1e-8)

        # if is_draw_pic:
        #     imageio.imwrite(f'{draw_result_dir}/{item_id}.jpg', show_im)

        # 每个图像的分数表
        item_score = {
            'id': item_id,
            'dice': dice,
        }

        each_item_score_list.append(item_score)

    # 总分计算
    all_score = {
        'dice': 0.,
    }

    for item_score in each_item_score_list:
        all_score['dice'] += item_score['dice']

    all_score['dice'] /= n_item

    text = yaml.safe_dump(all_score)
    print(text)
    open(result_path, 'w').write(text)

    open(each_result_path, 'w').write(yaml.safe_dump(each_item_score_list))
    print('complete')


def out_boge_data_to_mask():
    fold_id = 3
    boge_seg_data_path = f'boge_data/fold_{fold_id}_mask_pred.pkl'
    out_mask_path = f'boge_data/masks_{fold_id}.npy'

    boge_seg_data_list = pickle.load(open(boge_seg_data_path, 'rb'))

    '''
    - item - show_img_arr
           - d0-5 - [N,yx_conts]conts
                  - [N, 3]probs
    '''
    # 图像总数
    n_item = len(boge_seg_data_list)

    m = np.zeros([n_item, 256, 256, 5], np.int32)

    # 开始计算每幅图像分数
    for item_id, item in enumerate(boge_seg_data_list):
        print(f'process {item_id}')
        cur_pred_conts = contour_tool.tr_cv_to_my_contours(item)

        for c_id, cont in enumerate(cur_pred_conts):
            color_id = c_id + 1
            m[item_id, :, :, 0:1] = contour_tool.draw_contours(m[item_id, :, :, 0:1], [cont], (color_id,), -1)

    np.save(out_mask_path, m)
    print('complete')


if __name__ == '__main__':
    # test_cla()
    # test_seg()
    out_boge_data_to_mask()
