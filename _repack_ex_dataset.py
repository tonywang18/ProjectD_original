import numpy as np
import glob
import imageio
import cv2
import shutil
import os
import pickle
from lxml import etree as ET
import skimage.measure
from my_py_lib import contour_tool
import torch


def set1_label_to_points(xml_file):
    tree = ET.parse(xml_file)
    point_list = []
    for marker_type in tree.findall('.//Marker_Type/Marker'):     #find x,y,z coordinations
        f = []
        for x in marker_type.iter():
            f.append(x.text)
        loc = list(map(int,f[1:3]))
        point_list.append(loc)
    return point_list


def max_pool(im):
    im = torch.tensor(im, dtype=torch.float32)
    im = im[None, None]
    im = torch.nn.functional.max_pool2d(im, 5, stride=1, padding=2, dilation=1, ceil_mode=True)
    im = im[0, 0].numpy().astype(np.int)
    return im


def set2_label_to_points(im_path):
    im = imageio.imread(im_path)
    # hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS_FULL)
    # cv2.imshow('h', hls[..., 0])
    # cv2.imshow('l', hls[..., 1])
    # cv2.imshow('s', hls[..., 2])
    # h = hls[..., 0].astype(np.int)
    # h_max = max_pool(h)
    # h_min = -max_pool(-h)
    # new_h = np.where(h_max - h_min < 2, h, np.zeros_like(h))
    #
    # cv2.imshow('sad', new_h.astype(np.uint8))
    # cv2.imshow('im', im)

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h_v = hsv.transpose(2, 1, 0)[0].flatten()
    color = set(h_v) - set([174])  # find all color via vue value of HSV space
    cen = []
    for h in color:

        if h == 0:
            mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([0, 255, 255]))
            img_final = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.array((5, 5)))
            cnts, hier = cv2.findContours(img_final.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            center = []
            for each, cnt in enumerate(cnts):
                x, y, w, h = cv2.boundingRect(cnt)
                x_c, y_c = x + w / 2, y + h / 2
                center.append([x_c, y_c])
        else:
            lower = np.array([h - 2, 0, 0])
            upper = np.array([h + 2, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            img_final = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.array((5, 5)))
            cnts, hier = cv2.findContours(img_final.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            center = []
            for each, cnt in enumerate(cnts):
                x, y, w, h = cv2.boundingRect(cnt)
                x_c, y_c = x + w / 2, y + h / 2
                center.append([x_c, y_c])
        cen.extend(center)

    # for c in cen:
    #     im = cv2.circle(im, tuple(np.array(c, np.int)), 2, (65, 255, 128), -1)
    # cv2.imshow('asd', im[..., ::-1])
    # cv2.waitKey()

    return cen


def set3_label_to_points(im_path):
    im = imageio.imread(im_path)

    im_bin = (im > 128).astype(np.uint8)
    contours = contour_tool.find_contours(im_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, keep_invalid_contour=True)

    bboxes = [contour_tool.make_bbox_from_contour(i) for i in contours]
    pts = [[(b[2]+b[0])/2, (b[3]+b[1])/2] for b in bboxes]
    pts = np.asarray(pts, np.float32).reshape(-1, 2)[:, ::-1]

    # for c in pts:
    #     im = cv2.circle(im, tuple(np.array(c, np.int)), 3, (128,), -1)
    # cv2.imshow('asd', im[..., ::-1])
    # cv2.waitKey()

    return pts


if __name__ == '__main__':
    # dir1 = r'D:\bio-totem\project\projectD\extern_datasets\set1_label'
    # files = os.listdir(dir1)

    # new_files = [n.replace('_dots', '') for n in files]
    # for n, nn in zip(files, new_files):
    #     shutil.move(dir1+f'/{n}', dir1+f'/{nn}')

    dir1 = r'D:\bio-totem\project\projectD\extern_datasets\set3'
    dir1_label = r'D:\bio-totem\project\projectD\extern_datasets\set3_label'
    out_pkl = 'set3.pkl'
    out_data = []

    im_names = os.listdir(dir1)
    for im_name in im_names:
        basename = os.path.splitext(im_name)[0]
        label_name = f'{basename}.xml'

        im_path = f'{dir1}/{im_name}'
        label_path = f'{dir1_label}/{label_name}'

        # arr = set1_label_to_points(f'{dir1_label}/{basename}.xml')
        # arr = np.asarray(arr, np.float32).reshape(-1, 2).tolist()

        # arr = set2_label_to_points(f'{dir1_label}/{basename}.png')
        # arr = np.asarray(arr, np.float32).reshape(-1, 2).tolist()

        arr = set3_label_to_points(f'{dir1_label}/{basename}.png')
        arr = np.asarray(arr, np.float32).reshape(-1, 2).tolist()

        r, im_code = cv2.imencode('.png', imageio.imread(im_path)[..., ::-1])
        assert r

        # im = np.array(imageio.imread(im_path)[..., ::-1], order='C')
        # for c in arr:
        #     im = cv2.circle(im, tuple(np.array(c, np.int)), 3, (0,0,255), -1)
        # cv2.imshow('asd', im)
        # cv2.waitKey()

        item = {
            'basename': basename,
            'im_code': im_code,
            'label': arr,
        }

        out_data.append(item)

    pickle.dump(out_data, open(out_pkl, 'wb'))
