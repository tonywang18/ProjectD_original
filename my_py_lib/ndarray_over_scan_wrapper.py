'''
多维数组超采样工具
目前支持读取和写入

支持例如以下方式

ori_arr = np.zeros([20, 20])
arr = NdArrayOverScanWrapper(arr)

读取
不支持：
1.省略号
2.数字列表
支持：
1.单数字输入
2.None
3.切片
例如：
arr[None, :, 1]
arr[None, -5:5, 1]
arr[None, 8:12, -3]

写入
不支持：
1. 写入超出数组范围的设定值会被丢弃
2. 省略号
3. None
4. 数字列表
支持：
1. 单数字输入
2. 切片
例如：
arr[5:10, 3:4] = xxx
arr[-5:-1, 13] = xxx
注意：arr[1] 等同于 arr[1:2]
注意，写入时，填充的数组的维度数量要与原始数组一致。
注意，写入时，填充的数组的形状要与切片形状一致。

'''

import numpy as np
from typing import Sized


class NdArrayOverScanWrapper:
    def __init__(self, ori_arr: np.ndarray, pad_mode='constant', **pad_kwargs):
        self.data = ori_arr
        self.pad_mode = pad_mode
        self.pad_kwargs = pad_kwargs

    def __getitem__(self, ss):
        '''
        获取指定区域的数组
        :param ss:
        :return:
        '''
        if not isinstance(ss, Sized):
            ss = (ss,)

        cur_arr = self.data

        for i, v in enumerate(ss):
            if v is np.newaxis:
                cur_arr = np.expand_dims(cur_arr, i)

        assert len(ss) <= cur_arr.ndim

        for i, s in list(enumerate(ss))[::-1]:
            if isinstance(s, int) or (np.isscalar(s) and np.issubsctype(np.asarray(s), np.integer)):
                if s < 0:
                    p_before = abs(s)
                    pad_p = [[0, 0]] * cur_arr.ndim
                    pad_p[i] = [p_before, 0]
                    cur_arr = np.pad(cur_arr, pad_p, mode=self.pad_mode, **self.pad_kwargs)
                    cur_arr = np.take(cur_arr, s + p_before, axis=i)

                elif s >= cur_arr.shape[i]:
                    p_after = s - cur_arr.shape[i] + 1
                    pad_p = [[0, 0]] * cur_arr.ndim
                    pad_p[i] = [0, p_after]
                    cur_arr = np.pad(cur_arr, pad_p, mode=self.pad_mode, **self.pad_kwargs)
                    cur_arr = np.take(cur_arr, s, axis=i)

                else:
                    cur_arr = np.take(cur_arr, s, axis=i)

            elif isinstance(s, slice):
                p_before = 0
                p_after = 0

                if s.start is None:
                    s.start = 0
                elif s.start < 0:
                    p_before = abs(s.start)
                elif s.start >= cur_arr.shape[i]:
                    p_after = s.start - cur_arr.shape[i] + 1

                if s.stop is None:
                    s.stop = cur_arr.shape[i]
                elif s.stop < 0:
                    p_before = max(p_before, abs(s.stop))
                elif s.stop > cur_arr.shape[i]:
                    p_after = max(p_after, s.stop - cur_arr.shape[i])

                if p_before != 0 or p_after != 0:
                    pad_p = [[0, 0]] * cur_arr.ndim
                    pad_p[i] = [p_before, p_after]
                    cur_arr = np.pad(cur_arr, pad_p, mode=self.pad_mode, **self.pad_kwargs)

                s = np.arange(s.start + p_before, s.stop + p_before, s.step)

                cur_arr = np.take(cur_arr, s, axis=i)

            elif s is None:
                continue

            else:
                raise AssertionError('Error! Wrong index.')

        return cur_arr

    def __setitem__(self, ss, value):
        '''
        设定值。
        要加入多个限定条件，不然很难写或写不了
        1. 超出数组范围的设定值会被丢弃
        2. 不支持的省略号
        3. 不支持None
        4. 不支持数字列表

        :param key:
        :param value:
        :return:
        '''

        if not isinstance(ss, Sized):
            ss = (ss,)

        assert len(ss) <= self.data.ndim, 'Error! Too many dimensions are specified.'
        assert self.data.ndim == value.ndim, 'Error! Not support different ndim.'

        # 处理后的切片索引
        dss = []

        cur_value = value

        for dim_i, s in enumerate(ss):
            cur_ori_dim = self.data.shape[dim_i]
            cur_value_dim = cur_value.shape[dim_i]

            if isinstance(s, int) or (np.isscalar(s) and np.issubsctype(np.asarray(s), np.integer)):
                assert cur_value_dim == 1
                if s < 0:
                    dss.append(None)
                elif s >= cur_ori_dim:
                    dss.append(None)
                else:
                    assert cur_value_dim == 1
                    dss.append(slice(s, s + 1, 1))

            elif isinstance(s, slice):
                start = s.start
                stop = s.stop
                step = s.step

                if stop is None:
                    stop = cur_ori_dim
                if start is None:
                    start = 0

                # 仅支持1步长
                assert step is None or step == 1, 'Error! Only support step==1.'
                assert cur_value_dim == stop - start, 'Error! Value size is not equal slice.'

                if stop < 0:
                    stop = -1
                elif stop > cur_ori_dim:
                    cur_value = np.take(cur_value, np.arange(0, cur_value_dim-(stop-cur_ori_dim)), dim_i)
                    stop = cur_ori_dim

                cur_value_dim = cur_value.shape[dim_i]

                if start < 0:
                    if abs(start) > cur_value_dim:
                        start = -1
                    else:
                        cur_value = np.take(cur_value, np.arange(abs(start), cur_value_dim), dim_i)
                        start = 0
                elif start > cur_ori_dim:
                    start = -1

                if start == -1 or stop == -1 or stop - start < 1:
                    dss.append(None)
                else:
                    dss.append(slice(start, stop, step))

            # 因为语法 arr[[1,2,3],[1,2],[1,2,3]] 行为与预期不一致，取消该种方法
            # elif isinstance(s, Sized):
            #     new_s = []
            #     new_vs = []
            #     for i, iii in enumerate(s):
            #         if 0 <= iii < cur_ori_dim and i < cur_value_dim:
            #             new_s.append(iii)
            #             new_vs.append(i)
            #
            #     if len(new_s) == 0:
            #         dss.append(None)
            #     else:
            #         cur_value = np.take(cur_value, new_vs, dim_i)
            #         dss.append(new_s)

            else:
                raise AssertionError('Error! Wrong index.')

        for s in dss:
            if s is None:
                return

        dss = tuple(dss)
        self.data.__setitem__(dss, cur_value)


if __name__ == '__main__':
    a = np.random.randint(0, 1000, size=[4, 5, 6, 7])
    w = NdArrayOverScanWrapper(a, 'reflect')
    o = w[None, -1, -1:6, -1, -5:10]
    print(o.shape)

    # w[:, :, :, 1:5] = np.full([1,1,1,4], -1)

    a = np.full([15, 20], 0)
    w = NdArrayOverScanWrapper(a, 'reflect')
    w[-5:5, -5:5] = np.full([10, 10], 1)
    w[10:20, -5:5] = np.full([10, 10], 1)
    w[-5:5, 15:25] = np.full([10, 10], 1)
    w[10:20, 15:25] = np.full([10, 10], 1)
    print(w.data)
