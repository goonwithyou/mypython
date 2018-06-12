# @Time    : 2018/6/12 15:15
# @Author  : cap
# @FileName: myPipe.py
# @Software: PyCharm Community Edition

from multiprocessing import Pipe
from multiprocessing import Process
import os
import time


def func1(name, conn1, conn2):
    msg = 'hello' + str(name)
    conn2.send(msg)
    print('send:', msg)


def func2(_, conn1, conn2):
    time.sleep(2)
    print('recv:', conn1.recv())
    print('recv:', conn1.recv())
    print('recv:', conn1.recv())


def main():
    jobs = []
    conn1, conn2 = Pipe()
    funcs = [func1, func1, func1, func2]
    for i, func in enumerate(funcs):
        p = Process(target=func, args=(i, conn1, conn2))
        jobs.append(p)
        p.start()
    # print(jobs)
    for job in jobs:
        job.join()

    conn2.close()
    conn1.close()

if __name__ == '__main__':
    main()
