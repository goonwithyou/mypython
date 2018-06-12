# @Time    : 2018/6/12 16:17
# @Author  : cap
# @FileName: myQueue.py
# @Software: PyCharm Community Edition

from multiprocessing import Queue
from multiprocessing import Process
import time


def func_put(q):
    for i in range(5):
        q.put(i)


def func_get(q):
    for i in range(5):
        time.sleep(1)
        print(q.get())


def main():
    q = Queue(10)
    p1 = Process(target=func_put, args=(q,))
    p1.start()
    p2 = Process(target=func_get, args=(q,))
    p2.start()

    p1.join()
    q.close()
    p2.join()
    q.join_thread()


if __name__ == '__main__':
    main()
