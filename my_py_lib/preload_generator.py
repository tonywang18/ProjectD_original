import queue
import threading


def preload_generator(g, early_quit_flag=None, queue_size=10):
    '''
    一个预加载缓存队列，会自动预载数据
    :param g:
    :return:
    '''
    # 提前退出标志
    if early_quit_flag is None:
        early_quit_flag = [False]

    # 预载入线程
    def _run(g, q: queue.Queue, no_more_flag, early_quit_flag):
        for a in g:
            if early_quit_flag[0]:
                break
            q.put(a, block=True, timeout=None)
        no_more_flag[0] = True

    # 数据载入结束标志
    no_more_flag = [False]
    q = queue.Queue(maxsize=queue_size)
    th = threading.Thread(target=_run, args=(g, q, no_more_flag, early_quit_flag), daemon=True)
    th.start()
    while True:
        try:
            a = q.get(block=True, timeout=2)
            yield a
        except queue.Empty:
            if no_more_flag[0] and q.empty():
                break


if __name__ == '__main__':

    '''
    测试提前退出标志。当然可以不管，因为设定了守护线程，主线程退出时，守护线程也会退出，或者生成器被删除时，守护线程会自动异常退出
    流程：
    构造一个无限数字生成器
    使用预加载生成器修饰该生成器
    设定提前退出标志
    循环首
    当数字等于100时，设定提前退出标志
    循环尾
    程序退出
    '''

    def num_gen():
        i = 0
        while True:
            i += 1
            yield i

    g = num_gen()
    early_exit_flag = [False]
    g = preload_generator(g, early_exit_flag)

    for i in g:
        if i == 100:
            early_exit_flag[0] = True
        print(i)
