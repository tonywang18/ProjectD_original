import numpy as np
import pickle
import gzip
from my_py_lib import contour_tool
import os
import cv2
import imageio
import glob


def unpack(im_dir_path, mask_dir_path):
    im_names = os.listdir(im_dir_path)
    encode_ims = []
    labels = []
    names = []

    for im_name in im_names:
        print('Process', im_name, end=' ')

        im_basename = os.path.splitext(im_name)[0]
        im_data = open(f'{im_dir_path}/{im_basename}.png', 'rb').read()
        encode_ims.append(im_data)
        names.append(im_basename)

        label_arr = np.load(f'{mask_dir_path}/{im_basename}.npy')
        label_arr = label_arr.astype(np.int32)

        item_ids = np.unique(label_arr[..., 0])
        # 排除0
        item_ids = item_ids[item_ids != 0]

        cls_reg = {}
        for i in range(1, 8):
            cls_reg[i] = []

        ignore_count = 0

        for item_id in item_ids:
            bm = (label_arr[..., 0] == item_id).astype(np.uint8)
            _cons = contour_tool.find_contours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 排除超级小的轮廓
            if len(_cons) == 0:
                ignore_count += 1
                continue
            assert len(_cons) == 1
            cont = _cons[0]
            cls = label_arr[bm.astype(np.bool), 1]
            unq_c = np.unique(cls)
            assert len(unq_c) == 1
            c = unq_c[0]
            cls_reg[c].append(cont)
        labels.append(cls_reg)

        if (ignore_count > 0):
            print('ignore_too_small_cons_count', ignore_count)
        else:
            print()

    return encode_ims, labels, names


if __name__ == '__main__':
    ori_dataset_path = 'D:/Users/twd/Desktop/bio-totem/project/projectD/consep_dataset_ori/Test'

    im_dir_path = ori_dataset_path + '/Images'
    mask_dir_path = ori_dataset_path + '/Labels'
    encode_ims, labels, names = unpack(im_dir_path, mask_dir_path)
    assert len(encode_ims) == len(labels)
    all_data = {
        'encode_ims': encode_ims,
        'labels': labels,
        'names': names
    }
    open('D:/Users/twd/Desktop/bio-totem/project/projectD/consep_dataset_ori/test.pkl.gz', 'wb').write(gzip.compress(pickle.dumps(all_data)))


# if __name__ == '__main__':
#
#     color = {
#         0: (255, 0, 0),
#         1: (0, 255, 0),
#         2: (0, 0, 255),
#         3: (255, 128, 0),
#         4: (0, 128, 255),
#     }
#
#     d1 = np.load('example_images.npy')
#     d2 = np.load('example_mask.npy')
#
#     draw_ims = []
#     k = 1
#
#     for i in range(len(d1)):
#         im = d1[i].astype(np.uint8)
#         m = d2[i]
#         for d in range(5):
#             cids = np.unique(m)
#             cons = []
#             for c in cids:
#                 if c == 0:
#                     continue
#                 bm = (m[..., d] == c).astype(np.uint8)
#                 _cons = contour_tool.find_contours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 cons.extend(_cons)
#             im = contour_tool.draw_contours(im, cons, color[d], 2)
#         cv2.imshow('asd', im)
#         cv2.waitKey(1)
#         imageio.imwrite(f'{k}.png', im)
#         draw_ims.append(im)
#         k+=1
#
#     draw_ims = np.array(draw_ims)
#     np.savez_compressed('draw_ims.npz', draw_ims)
