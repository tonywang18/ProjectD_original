import numpy as np
import pickle
import gzip
from my_py_lib import contour_tool
import os
import cv2
import imageio


def unpack_img(im_array_path):
    arr = np.load(im_array_path).astype(np.uint8)
    encode_ims = []
    for i in range(len(arr)):
        encode_ims.append(cv2.imencode('.png', arr[i]))
    return encode_ims


def unpack_type(type_array_path):
    im_types = list(np.load(type_array_path))
    return im_types


def unpack_mask(mask_array_path):
    mask_array_path = os.path.abspath(mask_array_path)
    arr = np.load(mask_array_path).astype(np.int32)
    labels = []
    for i in range(len(arr)):
        cls_reg = {}
        m = arr[i]
        # 对通道5进行特殊处理，因为通道5背景是1，前景反而是0
        m[..., 5] = 1 - m[..., 5]
        for d in range(m.shape[-1]):
            cls_reg[d] = []
            con_ids = np.unique(m[..., d])
            cons = []
            for c in con_ids:
                if c == 0:
                    continue
                bm = (m[..., d] == c).astype(np.uint8)
                _cons = contour_tool.find_contours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cons.extend(_cons)
            cls_reg[d].extend(cons)
        labels.append(cls_reg)
    return labels


if __name__ == '__main__':
    im_arr_path = 'example_images.npy'
    type_arr_path = 'example_types.npy'
    mask_arr_path = 'example_mask.npy'
    encode_ims = unpack_img(im_arr_path)
    im_types = unpack_type(type_arr_path)
    labels = unpack_mask(mask_arr_path)
    assert len(encode_ims) == len(im_types) == len(labels)
    all_data = {
        'encode_ims': encode_ims,
        'im_types': im_types,
        'labels': labels
    }
    open('fold_1.pkl.gz', 'wb').write(gzip.compress(pickle.dumps(all_data)))


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
