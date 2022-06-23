import numpy as np
import cv2
import torch


def heatmap_nms(hm: np.ndarray):
    a = (hm * 255).astype(np.int32)
    a1 = cv2.blur(hm, (3, 3)).astype(np.int32)
    a2 = cv2.blur(hm, (5, 5)).astype(np.int32)
    a3 = cv2.blur(hm, (7, 7)).astype(np.int32)

    ohb = (hm > 0.).astype(np.float32)

    h = a + a1 + a2 + a3

    h = (h / 4).astype(np.float32)

    ht = torch.tensor(h)[None, None, ...]
    htm = torch.nn.functional.max_pool2d(ht, 9, stride=1, padding=4)
    hmax = htm[0, 0, ...].numpy()

    h = (h >= hmax).astype(np.float32)

    h = h * ohb

    h = cv2.dilate(h, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    return h


if __name__ == '__main__':
    import imageio

    om = imageio.imread(r"D:\Users\twd\Desktop\bio-totem\project\projectD\3_0.3_m7_64_det pred_circle_train_out_dir\img8_1det_a1_m.png")
    im = imageio.imread(r"D:\Users\twd\Desktop\bio-totem\project\projectD\3_0.3_m7_64_det pred_circle_valid_out_dir\img71_1det_a1_h.png")

    # cv2.imshow('om', om)
    # cv2.imshow('im', im)
    #
    # a1 = cv2.blur(im, (3, 3)).astype(np.int32)
    # a2 = cv2.blur(im, (5, 5)).astype(np.int32)
    # a3 = cv2.blur(im, (7, 7)).astype(np.int32)
    #
    # a = a1 + a2 + a3 + im
    #
    # h = a / 4
    #
    # # h = np.reshape(im, [100, 5, 100, 5]).astype(np.float32)
    # # h = np.sum(h, axis=(1, 3))
    # # h = (h / (h.max() + 1e-8) * 255)
    #
    # cv2.imwrite('sad1.png', h.astype(np.uint8))
    #
    # h = h.astype(np.uint8).astype(np.float32)
    #
    # ht = torch.tensor(h)[None, None, ...]
    # htm = torch.nn.functional.max_pool2d(ht, 3, stride=1, padding=1)
    # hmax = htm[0, 0, ...].numpy()
    #
    # h = np.where(h >= hmax, 1, 0)
    #
    # h = (h * 255).astype(np.uint8)

    h = heatmap_nms((im / 255))

    cv2.imshow('b3', h) #cv2.resize(h, (500, 500), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite('sad.png', (h * 255).astype(np.uint8))

    cv2.waitKey()


# if __name__ == '__main__':
#     import imageio
#
#     om = imageio.imread(r"D:\Users\twd\Desktop\bio-totem\project\projectD\3_0.3_m7_64_det pred_circle_train_out_dir\img3_1det_a1_m.png")
#     im = imageio.imread(r"D:\Users\twd\Desktop\bio-totem\project\projectD\3_0.3_m7_64_det pred_circle_train_out_dir\img3_1det_a1_h.png")
#
#     cv2.imshow('om', om)
#     cv2.imshow('im', im)
#
#     im_th = np.where(im > 0.2*255, im, 0)
#
#     cv2.imshow('im_th', im_th)
#
#     b1 = cv2.adaptiveThreshold(im_th, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 0)
#     k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     b1 = cv2.morphologyEx(b1, cv2.MORPH_DILATE, k)
#
#     cv2.imshow('b1', b1)
#
#     a1 = cv2.blur(im_th, (3, 3)).astype(np.int32)
#     a2 = cv2.blur(im_th, (5, 5)).astype(np.int32)
#     a3 = cv2.blur(im_th, (7, 7)).astype(np.int32)
#
#     im = im_th.astype(np.int32)
#
#     a = a1 + a2 + a3 + im
#
#     a = a / 4
#
#     a = np.asarray(a, np.uint8)
#
#     cv2.imshow('asd2', a)
#
#     b2 = cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 0)
#     cv2.imshow('b2', b2)
#
#     k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     b3 = cv2.morphologyEx(b2, cv2.MORPH_DILATE, k, iterations=3)
#
#     cv2.imshow('b3', b3)
#
#     cv2.waitKey()

