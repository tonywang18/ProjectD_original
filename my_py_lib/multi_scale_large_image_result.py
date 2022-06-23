'''
多尺度图块结果
可用于图像采样和保存热图结果
可以储存一个区域多个不同尺度的图像，每个尺度图像都有对应一张等大小的热图

大图扫描和结果类
与 MultiScalePatchResult 组合使用
从一个大图中，生成大量的MultiScalePatchResult进行分块处理，然后汇总，生成最终的大图结果
'''

import openslide
import cv2
import imageio
import numpy as np
from typing import Iterable, Callable
try:
    from . import contour_tool
    from . import list_tool
    from .multi_scale_patch_result import MultiScalePatchResult
    from .opsl_im_tool import opsl_read_region_any_ds
except ModuleNotFoundError:
    import contour_tool
    import list_tool
    from multi_scale_patch_result import MultiScalePatchResult
    from opsl_im_tool import opsl_read_region_any_ds


class MultiScaleLargeImageResult:
    '''
    大图结果类，将一张ndpi大图切成多个图块，然后使用 BigPicPatch 包装和生成每个图块的结果，然后再传回本类然后合并结果，得到大图结果
    '''
    def __init__(self, opsl_im: openslide.OpenSlide, n_class,
                 level_0_big_patch_hw=(5120, 5120), ds_factors=(1, 2, 4),
                 window_hw=(512, 512), level_0_roi_contours=None, *,
                 custom_final_contours_merge_pipe=None,
                 custom_patch_merge_pipe=None, custom_patch_postprocess_pipe=None,
                 custom_gen_contours_pipe=None, optim_contours=0,
                 result_dtype=np.float32, mask_dtype=np.uint8,
                 small_patch_step=0.5, big_patch_step=0.5):
        '''
        初始化
        :param opsl_im:                         输入openslide图像，或其他兼容openslide操作的图像类
        :param n_class:                         类别数量
        :param level_0_big_patch_hw:            大图块分辨率
        :param ds_factors:                      使用的下采样尺度比例
        :param window_hw:                       窗口高宽
        :param level_0_roi_contours:            0级图上，感兴趣区域的轮廓
        :param custom_final_contours_merge_pipe:自定义最终轮廓合并管道
        :param custom_patch_merge_pipe:         自定义图块合并管道
        :param custom_patch_postprocess_pipe:   自定义图块后处理通道管道，一般不使用。如果不为None，会在全部块合并完成后，执行一次
        :param custom_gen_contours_pipe:        自定义轮廓生成管道
        :param optim_contours:                  优化轮廓，当值大于0时，将会对轮廓进行优化
        :param result_dtype:                    指定大图块结果工具的热图结果的数据类型
        :param mask_dtype:                      指定大图块结果工具的热图掩码的数据类型
        :param small_patch_step:                指定大图块结果工具中小图块的滑窗步长
        :param big_patch_step:                  指定大图结果工具中大图块的滑窗步长
        '''
        if opsl_im is None:
            print('Warning! The opsl_im param is None, some functions will be disabled.')
        else:
            window_hw = np.int32(window_hw)
            level_0_big_patch_hw = np.int32(level_0_big_patch_hw)

        self.opsl_im = opsl_im
        self.n_class = n_class
        self.window_hw = window_hw
        self.level_0_big_patch_hw = level_0_big_patch_hw
        self.ds_factors = ds_factors
        self.custom_patch_merge_pipe = custom_patch_merge_pipe
        self.custom_patch_postprocess_pipe = custom_patch_postprocess_pipe
        self.custom_gen_contours_pipe = custom_gen_contours_pipe
        self.level_0_tissue_contours = level_0_roi_contours
        self.optim_contour = optim_contours
        self.result_dtype = result_dtype
        self.mask_dtype = mask_dtype
        self.small_patch_step = small_patch_step
        self.big_patch_step = big_patch_step

        # 自定义最终最终合并
        if isinstance(custom_final_contours_merge_pipe, Callable):
            self.custom_final_contours_merge_pipe = custom_final_contours_merge_pipe
        elif custom_final_contours_merge_pipe is None:
            self.custom_final_contours_merge_pipe = self._default_final_contours_merge_pipe
        else:
            raise AssertionError('Error! Bad param custom_final_contours_merge_pipe.')

        if opsl_im is None:
            self.ds_hw = None
        else:
            ds_hw = []
            for factor in ds_factors:
                rescale_patch_hw = np.int32(level_0_big_patch_hw / factor)
                if np.any(level_0_big_patch_hw % rescale_patch_hw != 0):
                    print('warning: Recommended big_patch_hw is a multiple of each ds_factor')
                ds_hw.append(rescale_patch_hw)
            self.ds_hw = ds_hw
        self.all_contours = []
        self.all_classes = []

    def next_ms_patch_gen(self):
        '''
        大图块结果工具的生成器
        该函数与update_ms_patch函数成配对关系
        :return:
        '''
        assert self.opsl_im is not None, 'Error! opsl_im is None.'

        top_im_hw = np.array(self.opsl_im.level_dimensions[0][::-1])
        y_list = range(0, top_im_hw[0], int(self.level_0_big_patch_hw[0] * self.big_patch_step))
        x_list = range(0, top_im_hw[1], int(self.level_0_big_patch_hw[1] * self.big_patch_step))
        for y in y_list:
            for x in x_list:
                level_0_patch_start_yx = (y, x)
                multi_scale_img = []
                # 检测 level_0_tissue_contours，若存在则可以提前判定是否跳过
                if self.level_0_tissue_contours is not None:
                    bbox = np.asarray([y, x, y + self.level_0_big_patch_hw[0], x + self.level_0_big_patch_hw[1]])
                    bbox_cont = contour_tool.make_contour_from_bbox(bbox)
                    ious = contour_tool.calc_iou_with_contours_1toN(bbox_cont, self.level_0_tissue_contours)
                    if np.max(ious) == 0:
                        continue

                for factor in self.ds_factors:
                    a = opsl_read_region_any_ds(self.opsl_im, factor, level_0_patch_start_yx, self.level_0_big_patch_hw, 0)
                    multi_scale_img.append(a)

                bpp = MultiScalePatchResult(self.n_class, multi_scale_img, level_0_patch_start_yx,
                                            self.window_hw, self.level_0_big_patch_hw,
                                            level_0_tissue_contours=self.level_0_tissue_contours,
                                            custom_patch_merge_pipe=self.custom_patch_merge_pipe,
                                            custom_patch_postprocess_pipe=self.custom_patch_postprocess_pipe,
                                            custom_gen_contours_pipe=self.custom_gen_contours_pipe,
                                            result_dtype=self.result_dtype, mask_dtype=self.mask_dtype,
                                            patch_step=self.small_patch_step)
                yield bpp

    def update_ms_patch(self, bpp: MultiScalePatchResult, *, return_contours=False):
        '''
        收集大图块结果工具的输出
        该函数与next_ms_patch_gen函数成配对关系
        :param bpp:
        :param return_contours: 是否返回收集到的轮廓
        :return:
        '''
        conts, clss = bpp.get_contours()
        if self.optim_contour > 0:
            conts = contour_tool.simple_contours(conts, self.optim_contour)

        valids = contour_tool.is_valid_contours(conts)
        conts = list_tool.list_multi_get_with_bool(conts, valids)
        clss = list_tool.list_multi_get_with_bool(clss, valids)

        self.all_contours.extend(conts)
        self.all_classes.extend(clss)

        if return_contours:
            return conts, clss

    def get_contours(self):
        '''
        获得最终的轮廓和类别
        :return:
        '''
        new_conts, new_clss = self.custom_final_contours_merge_pipe(self, self.all_contours, self.all_classes)
        return new_conts, new_clss

    @staticmethod
    def _default_final_contours_merge_pipe(self, all_contours, all_classes):
        '''
        默认最终轮廓合并管道
        :param self:            大图结果合成工具自身，一般不使用
        :param all_contours:    大图结果合成工具收集的所有轮廓
        :param all_classes:     大图结果合成工具收集的所有类别
        :return:
        '''
        new_conts = []
        new_clss = []

        d = list_tool.list_group_by_classes(all_contours, all_classes)
        for cls, conts in d.items():
            conts = contour_tool.merge_multi_contours(conts)
            new_conts.extend(conts)
            new_clss.extend([cls] * len(conts))

        return new_conts, new_clss


if __name__ == '__main__':
    import time

    aaa = time.time()

    im = openslide.open_slide("./dataset/ims/#2.ndpi")
    bpr = MultiScaleLargeImageResult(im, 3, [5120, 5120], ds_factors=[1, 2, 4])

    # from my_py_lib.preload_generator import preload_generator

    bpp_g = bpr.next_ms_patch_gen()

    ccc = 0

    for bpp in bpp_g:
        bpp: MultiScalePatchResult

        for batch_info, batch_im in bpp.batch_get_im_patch_gen(15):
            print(batch_info[0])
            new_batch_im = []
            for im in batch_im:
                im2 = np.zeros([*im.shape[:2], 3], dtype=np.float32)
                im2[:, :, 0] = 0.5
                # im2[:, :, 1] = 1
                im2 = cv2.circle(im2, (0, 0), 120, (0, 1, 0), 5)
                im2 = cv2.circle(im2, (256, 256), 60, (0, 0, 1), 5)
                new_batch_im.append(im2)
            bpp.batch_update_result(batch_info, new_batch_im)
        bpr.update_ms_patch(bpp)
        if ccc == 0:
            imageio.imwrite('a1.png', (bpp.multi_scale_result[0].data * 255).astype(np.uint8))
            imageio.imwrite('a2.png', (bpp.multi_scale_result[1].data * 255).astype(np.uint8))
            imageio.imwrite('a3.png', (bpp.multi_scale_result[2].data * 255).astype(np.uint8))
            imageio.imwrite('b1.png', bpp.multi_scale_img[0].data)
            imageio.imwrite('b2.png', bpp.multi_scale_img[1].data)
            imageio.imwrite('b3.png', bpp.multi_scale_img[2].data)

        ccc += 1
        print(ccc)
        if ccc > 3:
            break

    cons, clss = bpr.get_contours()

    a = np.zeros([5120, 5120, 1], np.uint8)
    a = contour_tool.draw_contours(a, cons, 255, 10)
    imageio.imwrite('a.jpg', a)

    print(time.time() - aaa)
