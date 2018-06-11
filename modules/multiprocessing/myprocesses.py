# @Time    : 2018/6/11 17:38
# @Author  : cap
# @FileName: myprocesses.py
# @Software: PyCharm Community Edition

import multiprocessing as mp
import time
import os


def func1():
    time.sleep(2)
    print('eating')
    print(os.getppid(), '-->', os.getpid())


def func2():
    time.sleep(4)
    print('playing')
    print(os.getppid(), '-->', os.getpid())


def func3():
    time.sleep(3)
    print('sleeping')
    print(os.getppid(), '-->', os.getpid())


def main():
    funcs = [func1, func2, func3]
    processes = []
    for func in funcs:
        p = mp.Process(target=func)
        processes.append(p)
        p.start()

    for process in processes:
        process.join()


if __name__ == '__main__':
    main()
