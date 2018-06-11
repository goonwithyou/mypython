# @Time    : 2018/6/11 17:28
# @Author  : cap
# @FileName: myprocess.py
# @Software: PyCharm Community Edition
import multiprocessing as mp
import os


def func():
    print('the event of child process')
    print('parent pid', os.getppid())


def main():
    print('main process pid:', os.getpid())
    p = mp.Process(target=func)
    p.start()
    p.join()


if __name__ == '__main__':
    main()
