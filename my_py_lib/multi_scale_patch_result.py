'''
多尺度图块结果
可用于图像采样和保存热图结果
可以储存一个区域多个不同尺度的图像，每个尺度图像都有对应一张等大小的热图
'''

import cv2
import numpy as np
from typing import Iterable, Callable
from .coords_over_scan_gen import n_step_scan_coords_gen_v2
from .image_over_scan_wrapper import ImageOverScanWrapper
from .universal_batch_generator import universal_batch_generator
from . import contour_tool
from . import im_tool


class MultiScalePatchResult:
    '''
    现在对这个进行简化，仅负责图像的管道处理和储存
    大图块就是这里
    现在加入缓存机制，重复读取时将会使用缓存
    '''
    def __init__(self, n_class, multi_scale_img, top_left_point, window_hw, level_0_patch_hw,
                 ignore_contour_near_border_ratio=0.001, ignore_patch_near_border_ratio=0.1,
                 level_0_tissue_contours=None, *,
                 multi_scale_result=None, multi_scale_mask=None,
                 custom_patch_merge_pipe=None, custom_patch_postprocess_pipe=None,
                 custom_gen_contours_pipe=None,
                 result_dtype=np.float32, mask_dtype=np.uint8,
                 patch_step=0.5):
        '''
        :param n_class:                             热图储存的类别数量
        :param multi_scale_img:                     多个尺度的图像，# 后半句已过时 注意，以第一个图像为基准，其他的图都会以它进行缩放
        :param top_left_point:                      此区域的左上角坐标，yx格式
        :param window_hw:                           滑窗大小
        :param level_0_patch_hw:                    图块的原始大小，当多个尺度图像不包含0级图时，可以缩放到0级图尺度
        :param ignore_contour_near_border_ratio:    忽略靠近边缘的轮廓，如果轮廓与边缘的最小距离小于该比例，则抛弃该轮廓
        :param ignore_patch_near_border_ratio:      忽略每一个小滑窗patch的边缘比例
        :param level_0_tissue_contours:             level_0尺度下的mask轮廓，可以加速处理，不在轮廓内的滑窗将会被跳过
        :param multi_scale_result:                  用来恢复结果缓存
        :param multi_scale_mask:                    用来恢复mask缓存
        :param custom_patch_merge_pipe:             自定义滑窗合并方法，如不设定，self._default_patch_merge是默认的处理方法
        :param custom_patch_postprocess_pipe:       自定义滑窗合并方法，如不设定，self._default_patch_postprocess是默认的处理方法，另外，该方法需要手动调用
        :param custom_gen_contours_pipe:            自定义轮廓生成方法，如不设定，self._default_get_contours是默认的处理方法
        :param result_dtype:                        结果的数据类型，默认为 np.float32
        :param mask_dtype:                          掩码的数据类型，默认为 np.uint8
        :param patch_step:                          滑块步长，默认为0.5，也就是半步长
        '''
        self.mask_dtype = mask_dtype
        self.result_dtype = result_dtype
        self.multi_scale_img = [ImageOverScanWrapper(im) for im in multi_scale_img]
        self.top_left_point = np.int32(top_left_point)
        self.n_class = n_class
        self.window_hw = window_hw
        self.level_0_patch_hw = np.int32(level_0_patch_hw)
        self.ignore_contour_near_border_ratio = ignore_contour_near_border_ratio
        self.ignore_patch_ratio = ignore_patch_near_border_ratio
        self.tissue_contours = level_0_tissue_contours
        self.patch_step = patch_step

        # 缓存标记
        self._contours_cache_data = None
        self._contours_need_update = True
        self._need_heatmap_fresh = True
        self._cache_heatmap = None

        # 多尺度结果
        if multi_scale_result is None:
            multi_scale_result = [np.zeros([*im.shape[:2], n_class], self.result_dtype) for im in multi_scale_img]
        else:
            for i, _ in enumerate(multi_scale_img):
                H, W, C = multi_scale_result[i].shape
                assert H == multi_scale_img[i].shape[0] and W == multi_scale_img[i].shape[1] and C == n_class \
                       and multi_scale_result[i].dtype == self.result_dtype
        self.multi_scale_result = [ImageOverScanWrapper(im) for im in multi_scale_result]

        # 多尺度掩码，用来记录每一副图的每个部分是否被处理过，目前只用于自定义patch融合管道和自定义轮廓生成管道，默认管道不使用
        if multi_scale_mask is None:
            multi_scale_mask = [np.zeros([*im.shape[:2], 1], self.mask_dtype) for im in multi_scale_img]
        else:
            for i, _ in enumerate(multi_scale_mask):
                H, W, C = multi_scale_mask[i].shape
                assert H == multi_scale_img[i].shape[0] and W == multi_scale_img[i].shape[1] and C == 1 \
                       and multi_scale_mask[i].dtype == self.mask_dtype
        self.multi_scale_mask = [ImageOverScanWrapper(im) for im in multi_scale_mask]

        # 检查和设定自定义方法
        assert custom_patch_merge_pipe is None or isinstance(custom_patch_merge_pipe, Callable)
        assert custom_patch_postprocess_pipe is None or isinstance(custom_patch_postprocess_pipe, Callable)
        assert custom_gen_contours_pipe is None or isinstance(custom_gen_contours_pipe, Callable)
        self.custom_patch_merge_pipe = custom_patch_merge_pipe
        self.custom_patch_postprocess_pipe = custom_patch_postprocess_pipe
        self.custom_gen_contours_pipe = custom_gen_contours_pipe

    def _get_pic_patch(self, level, yx_start, yx_end, pad_value):
        '''
        获得图区内某个图块
        :param level:       尺度号
        :param yx_start:    左上角起始坐标
        :param yx_end:      右下角起始坐标
        :param pad_value:   过采样区域填充的值
        :return:
        '''
        out_im = self.multi_scale_img[level].get(yx_start, yx_end, pad_value)
        return out_im

    def _set_pic_patch(self, level, yx_start, yx_end, patch_result_new):
        '''
        根据尺度信息和坐标，将输入图填充到指定区域
        :param level:
        :param yx_start:
        :param yx_end:
        :param patch_result_new:
        :return:
        '''
        # 更新缓存标记
        self._contours_need_update = True
        self._need_heatmap_fresh = True
        patch_result_cur = self.multi_scale_result[level].get(yx_start, yx_end)
        patch_mask_cur = self.multi_scale_mask[level].get(yx_start, yx_end)
        assert patch_result_new.ndim == patch_result_cur.ndim and patch_result_new.shape[-1] == patch_result_cur.shape[-1], 'Error! New patch size is unexcept.'
        assert patch_result_new.shape[0] == yx_end[0] - yx_start[0] and patch_result_new.shape[1] == yx_end[1] - yx_start[1], 'Error! New patch size is unexcept.'

        # 处理忽略边缘问题
        ignore_patch_ratio = np.clip(self.ignore_patch_ratio, 0, 1.)
        # 根据 ignore_patch_ratio 参数，使区域更小，从而忽略掉边缘的值
        ignore_edge_hw_pixel = (np.array(patch_result_new.shape[:2]) * ignore_patch_ratio / 2).astype(np.int32)
        ys = ignore_edge_hw_pixel[0]
        ye = None if ignore_edge_hw_pixel[0] == 0 else -ignore_edge_hw_pixel[0]
        xs = ignore_edge_hw_pixel[1]
        xe = None if ignore_edge_hw_pixel[1] == 0 else -ignore_edge_hw_pixel[1]

        if self.custom_patch_merge_pipe is None:
            new_result, new_mask = self._default_patch_merge(self, patch_result_cur, patch_result_new, patch_mask_cur)
        else:
            new_result, new_mask = self.custom_patch_merge_pipe(self, patch_result_cur, patch_result_new, patch_mask_cur)

        patch_result_cur[ys: ye, xs: xe] = new_result[ys: ye, xs: xe]
        patch_mask_cur[ys: ye, xs: xe] = new_mask[ys: ye, xs: xe]

        self.multi_scale_result[level].set(yx_start, yx_end, patch_result_cur)
        self.multi_scale_mask[level].set(yx_start, yx_end, patch_mask_cur)

    def get_im_patch_gen(self, *, levels: list=None):
        '''
        获得一个图块生成器，
        :param levels:
        :return:
        '''
        for level, im in enumerate(self.multi_scale_img):
            if levels is not None:
                if level not in levels:
                    continue
            for yx_start, yx_end in n_step_scan_coords_gen_v2(im.shape[:2], self.window_hw, self.patch_step):
                # 这里代码应该没问题了
                # 如果全局轮廓已定义，则可以预先判断，可以跳过轮廓外区域的运算
                if self.tissue_contours is not None:
                    bbox = np.asarray([*yx_start, yx_start[0] + self.window_hw[0], yx_start[1] + self.window_hw[1]])
                    bbox_cont = contour_tool.make_contour_from_bbox(bbox)
                    factor_hw = self.level_0_patch_hw / im.shape[:2]
                    bbox_cont = contour_tool.resize_contours([bbox_cont], factor_hw)[0]
                    bbox_cont = contour_tool.offset_contours([bbox_cont], (0, 0), self.top_left_point)[0]
                    ious = contour_tool.calc_iou_with_contours_1toN(bbox_cont, self.tissue_contours)
                    if np.max(ious) == 0:
                        continue

                out_im = self._get_pic_patch(level, yx_start, yx_end, 0)
                # # 第二个加速功能，判断图像白色像素（3通道均大于215的像素）占比是否大于95%，若大于则跳过
                # bm1 = np.all(out_im > np.reshape([215, 215, 215], [1, 1, 3]), -1)
                # if bm1.sum(dtype=np.int) * 0.95 > bm1.shape[0] * bm1.shape[1]:
                #     continue

                info = (level, yx_start, yx_end)
                yield info, out_im

    def update_result(self, info, result_im):
        '''
        更新结果和掩码
        :param info:
        :param result_im:
        :return:
        '''
        level, yx_start, yx_end = info
        assert result_im.shape[0] == self.window_hw[0] and result_im.shape[1] == self.window_hw[1]
        self._set_pic_patch(level, yx_start, yx_end, result_im)

    def batch_get_im_patch_gen(self, batch_size, *, levels=None):
        '''
        为 get_pic_gen 的批量版本，可以用来加速
        :param batch_size:
        :param levels:
        :return:
        '''
        g = self.get_im_patch_gen(levels=levels)
        return universal_batch_generator(g, batch_size)

    def batch_update_result(self, batch_info, batch_result):
        '''
        为update_result的批量版本，批量更新结果和掩码
        :param batch_info:
        :param batch_result:
        :return:
        '''
        assert len(batch_info) == len(batch_result)
        for p in zip(batch_info, batch_result):
            self.update_result(*p)

    def do_patch_postprocess(self):
        '''
        此时已多尺度图像已生成完成，这里可以对多尺度图像进行总的后处理操作
        默认为什么也不干
        :return:
        '''
        if self.custom_patch_postprocess_pipe is None:
            func = self._default_patch_postprocess
        else:
            func = self.custom_patch_postprocess_pipe

        multi_scale_img = [d.data for d in self.multi_scale_img]
        multi_scale_result = [d.data for d in self.multi_scale_result]
        multi_scale_mask = [d.data for d in self.multi_scale_mask]

        multi_scale_img, multi_scale_result, multi_scale_mask =\
            func(self, multi_scale_img, multi_scale_result, multi_scale_mask)

        self.multi_scale_img = [ImageOverScanWrapper(d) for d in multi_scale_img]
        self.multi_scale_result = [ImageOverScanWrapper(d) for d in multi_scale_result]
        self.multi_scale_mask = [ImageOverScanWrapper(d) for d in multi_scale_mask]

    @staticmethod
    def _default_patch_merge(self, patch_result, patch_result_new, patch_mask):
        '''
        默认的滑窗图合并函数，合并时取最大值
        :param self:                引用大图块自身，用于实现某些特殊用途，一般不使用
        :param patch_result:        当前滑窗区域的结果
        :param patch_result_new:    新的滑窗区域的结果
        :param patch_mask:          当前掩码，用于特殊用途，这里不使用
        :return: 返回合并后结果和更新的掩码
        '''
        new_result = np.maximum(patch_result, patch_result_new)
        new_mask = patch_mask
        return new_result, new_mask

    @staticmethod
    def _default_patch_postprocess(self, multi_scale_img, multi_scale_result, multi_scale_mask):
        '''
        默认的大图块后处理函数，什么都不干
        :param self:                引用大图块自身，用于实现某些特殊用途，一般不使用
        :param multi_scale_img:     多尺度原图
        :param multi_scale_result:  多尺度结果图
        :param multi_scale_mask:    多尺度mask图
        :return: 返回处理后的多尺度热图和多尺度mask
        '''
        return multi_scale_img, multi_scale_result, multi_scale_mask

    @staticmethod
    def _default_gen_contours(self, multi_scale_result, multi_scale_mask):
        '''
        默认的轮廓生成函数，当前只是简单合并热图，然后以0.5做阈值，然后生成轮廓
        :param self:                引用大图块自身，用于实现某些特殊用途，一般不使用
        :param multi_scale_result:  多尺度结果图
        :param multi_scale_mask:    多尺度mask图
        :return: 返回轮廓和每个轮廓的类别
        '''
        prob = 0.5
        top_result = multi_scale_result[0]
        top_result_hw = top_result.shape[:2]

        for ms in multi_scale_result[1:]:
            ms = im_tool.resize_image(ms, top_result.shape[:2], interpolation=cv2.INTER_LINEAR)
            top_result = np.maximum(top_result, ms)
        contours = []
        clss = []
        top_result = (top_result > prob).astype(np.uint8)
        for c in range(top_result.shape[2]):
            cs = contour_tool.find_contours(top_result[:, :, c], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(cs)
            clss.extend([c] * len(cs))
        return top_result_hw, contours, clss

    def get_contours(self, dont_use_start_yx=False, dont_resize_contours=False):
        '''
        获取轮廓
        :param dont_use_start_yx:       若为True，则返回的轮廓坐标不会偏移
        :param dont_resize_contours:    若为True，则返回的轮廓不会缩放到level_0尺度
        :return:
        '''
        if self._contours_need_update:
            multi_scale_result = [d.data for d in self.multi_scale_result]
            multi_scale_mask = [d.data for d in self.multi_scale_mask]
            if self.custom_gen_contours_pipe is None:
                top_result_hw, contours, clss = self._default_gen_contours(self, multi_scale_result, multi_scale_mask)
            else:
                top_result_hw, contours, clss = self.custom_gen_contours_pipe(self, multi_scale_result, multi_scale_mask)

            need_remove_ids = contour_tool.is_contours_too_close_border(contours, top_result_hw, self.ignore_contour_near_border_ratio)
            for i in sorted(need_remove_ids)[::-1]:
                del contours[i]
                del clss[i]

            self._contours_cache_data = [top_result_hw, contours, clss]
            self._contours_need_update = False
        else:
            top_result_hw, contours, clss = self._contours_cache_data

        top_result_hw = np.asarray(top_result_hw)

        # 缩放轮廓尺寸为0级尺寸
        if not dont_resize_contours:
            if np.any(top_result_hw != self.level_0_patch_hw):
                contours = contour_tool.resize_contours(contours, self.level_0_patch_hw / top_result_hw)

        # 偏移轮廓
        if not dont_use_start_yx:
            contours = contour_tool.offset_contours(contours, (0, 0), self.top_left_point)

        return contours, clss


if __name__ == '__main__':
    ims = [np.full([s, s, 3], 128, np.uint8) for s in [5120, 2560, 1280, 640]]
    p = MultiScalePatchResult(3, ims, [100, 100], [512, 512], [10240, 10240])

    for batch_info, batch_im in p.batch_get_im_patch_gen(10, levels=None):
        print(batch_info)
        o = np.zeros([len(batch_info), 512, 512, 3], np.float32)
        p.batch_update_result(batch_info, o)
