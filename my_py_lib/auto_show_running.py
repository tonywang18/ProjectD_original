'''
运行中显示工具
可以动态看到训练时生成的图像
'''


import threading
import cv2
import numpy as np
try:
    from im_tool import pad_picture, ensure_image_has_3dim
except ModuleNotFoundError:
    from .im_tool import pad_picture, ensure_image_has_3dim


class AutoShowRunning:

    def __init__(self, out_hw=(428, 1280), show_num_hw=(2, 6), pad_value=128, title='running show'):
        '''
        初始化
        :param out_hw: 显示的图像大小
        :param show_num_hw: 每列和每行显示的图像数量
        '''
        self._swap_lock = threading.Lock()
        self._im_list = []
        self.out_hw = out_hw
        self.show_num_hw = show_num_hw
        self.each_im_hw = (out_hw[0] // show_num_hw[0], out_hw[1] // show_num_hw[1])
        self.show_num = np.prod(show_num_hw)
        self._need_quit = False
        self.pad_value = pad_value
        self.title = title

        self._show_thread = threading.Thread(target=self._show_run, daemon=True)
        self._show_thread.start()

    def show_im(self, batch_im):
        if self._swap_lock.acquire(blocking=True):
            self._im_list.clear()
            for i in range(min(self.show_num, len(batch_im))):
                self._im_list.append(batch_im[i])
            self._swap_lock.release()

    def destroy(self):
        self._need_quit = True
        cv2.destroyWindow(self.title)
        cv2.waitKey(10)

    def _show_run(self):
        while not self._need_quit:
            big_im = np.empty([*self.out_hw, 3], np.uint8)
            big_im[..., :] = self.pad_value
            if self._swap_lock.acquire(blocking=True):
                for id, im in enumerate(self._im_list):
                    im = pad_picture(im, self.each_im_hw[1], self.each_im_hw[0], cv2.INTER_AREA, fill_value=self.pad_value)
                    im = ensure_image_has_3dim(im)
                    if im.shape[2] > 3:
                        im = im[:, :, :3]
                    elif im.shape[2] == 2:
                        _pad = np.zeros([im.shape[0], im.shape[1], 1], dtype=im.dtype)
                        im = np.concatenate([im, _pad], axis=2)
                    h = id // self.show_num_hw[1]
                    w = id %  self.show_num_hw[1]
                    big_im[h*self.each_im_hw[0]: (h+1)*self.each_im_hw[0], w*self.each_im_hw[1]: (w+1)*self.each_im_hw[1]] = im
                self._swap_lock.release()

                cv2.imshow(self.title, big_im[..., ::-1])
                cv2.waitKey(5000)


if __name__ == '__main__':
    import time

    sr = AutoShowRunning()

    while True:
        batch_im = np.random.randint(0, 255, [15, 512, 512, 3], np.uint8)
        sr.show_im(batch_im)
        time.sleep(2)
