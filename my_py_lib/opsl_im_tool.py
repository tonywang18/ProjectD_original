'''
openslide image tools
and
tiffslide image tools
'''



import numpy as np
import cv2
from typing import Union, Tuple
try:
    from . import im_tool
    from .numpy_tool import round_int
    from . import image_over_scan_wrapper
    from . import coords_over_scan_gen
    from .void_cls import VC
except (ModuleNotFoundError, ImportError):
    from my_py_lib import im_tool
    from my_py_lib.numpy_tool import round_int
    from my_py_lib import image_over_scan_wrapper
    from my_py_lib import coords_over_scan_gen
    from void_cls import VC

try:
    import openslide as opsl
except (ModuleNotFoundError, ImportError):
    opsl = VC()
    opsl.OpenSlide = None

try:
    import tiffslide as tisl
except (ModuleNotFoundError, ImportError):
    tisl = VC()
    tisl.TiffSlide = None


def read_region_any_ds(opsl_im: Union[opsl.OpenSlide, tisl.TiffSlide],
                       ds_factor: float,
                       level_0_start_yx: Tuple[int, int],
                       level_0_region_hw: Tuple[int, int],
                       close_thresh: float=0.0):
    '''
    从多分辨率图中读取任意尺度图像
    :param opsl_im:             待读取的 OpenSlide 或 tisl.TiffSlide 图像
    :param ds_factor:           下采样尺度
    :param level_0_start_yx:    所读取区域在尺度0上的位置
    :param level_0_region_hw:   所读取区域在尺度0上的高宽
    :param close_thresh:        如果找到足够接近的下采样尺度，则直接使用最接近的，不再对图像进行缩放，默认为0，即为关闭
    :return:
    '''
    level_downsamples = opsl_im.level_downsamples

    level_0_start_yx = round_int(level_0_start_yx)
    level_0_region_hw = round_int(level_0_region_hw)

    assert ds_factor > 0, f'Error! Not allow ds_factor <= 0. ds_factor={ds_factor}'

    # base_level = None
    # ori_patch_hw = None
    # 这里需要使用round，因为大小需要比较准确，避免 511.99 被压到 511
    target_patch_hw = round_int(level_0_region_hw / ds_factor)

    is_close_list = np.isclose(ds_factor, level_downsamples, rtol=close_thresh, atol=0)
    if np.any(is_close_list):
        # 如果有足够接近的
        level = np.argmax(is_close_list)
        base_level = level
        ori_patch_hw = target_patch_hw
    else:
        # 没有足够接近的，则寻找最接近，并且分辨率更高的，然后再缩放。
        # 增加ds_factor超过level_downsamples边界的支持
        if ds_factor > max(opsl_im.level_downsamples):
            # 如果ds_factor大于图像自身包含的最大的倍率
            level = opsl_im.level_count - 1
        elif ds_factor < min(opsl_im.level_downsamples):
            # 如果ds_factor小于图像自身最小的倍率
            level = 0
        else:
            level = np.argmax(ds_factor < np.array(opsl_im.level_downsamples)) - 1
        assert level >= 0, 'Error! read_im_mod found unknow level {}'.format(level)
        base_level = level
        level_ds_factor = level_downsamples[level]
        ori_patch_hw = np.array(target_patch_hw / level_ds_factor * ds_factor, np.int64)

    # 读取图块，如果不是目标大小则缩放到目标大小
    # 使用 int64 避免 tiffslide 坐标溢出导致的错位的问题
    im = np.uint8(opsl_im.read_region(np.int64(level_0_start_yx[::-1]), base_level, np.int64(ori_patch_hw[::-1])))[:, :, :3]
    if np.any(ori_patch_hw != target_patch_hw):
        im = im_tool.resize_image(im, target_patch_hw, cv2.INTER_AREA)
    return im


opsl_read_region_any_ds = read_region_any_ds


def make_thumb_any_level(bim: Union[opsl.OpenSlide, tisl.TiffSlide],
                         ds_level=0,
                         thumb_size=2048,
                         tile_hw=(512, 512),
                         ):
    '''
    从任意层级创建缩略图
    :param bim:         大图像，允许为 OpenSlide 或 TiffSlide
    :param ds_level:    指定层级
    :param thumb_size:  缩略图最长边的长度
    :param tile_hw:     采样时图块大小
    :return:
    '''
    lv0_hw = bim.level_dimensions[0][::-1]

    level_hw = bim.level_dimensions[ds_level][::-1]

    to_lv0_factor = lv0_hw / np.float32(level_hw)
    factor = np.min(thumb_size / np.float32(level_hw))

    thumb_hw = round_int(np.float32(level_hw) * factor)

    thumb_im = np.zeros([*thumb_hw, 3], np.uint8)
    thumb_im_wrap = image_over_scan_wrapper.ImageOverScanWrapper(thumb_im)

    for yx_start, yx_end in coords_over_scan_gen.n_step_scan_coords_gen_v2(level_hw, window_hw=tile_hw, n_step=1):

        lv0_pos = round_int(yx_start[::-1] * to_lv0_factor)

        tile = np.asarray(bim.read_region(lv0_pos, ds_level, tile_hw[::-1]))[..., :3]

        yx_start = round_int(np.float32(yx_start) * factor)
        yx_end = round_int(np.float32(yx_end) * factor)

        new_hw = yx_end - yx_start

        try:
            tile = im_tool.resize_image(tile, new_hw, interpolation=cv2.INTER_AREA)
        except:
            continue

        thumb_im_wrap.set(yx_start, yx_end, tile)

    return thumb_im



if __name__ == '__main__':
    import imageio

    opsl_im = opsl.OpenSlide(r"#142.ndpi")
    im = opsl_read_region_any_ds(opsl_im, ds_factor=4, level_0_start_yx = (20000, 20000), level_0_region_hw = (5120, 5120))
    imageio.imwrite('1.jpg', im)

    opsl_im = tisl.TiffSlide(r"#142.ndpi")
    im = opsl_read_region_any_ds(opsl_im, ds_factor=4, level_0_start_yx = (20000, 20000), level_0_region_hw = (5120, 5120))
    imageio.imwrite('2.jpg', im)
