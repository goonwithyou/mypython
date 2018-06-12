# @Time    : 2018/6/12 10:21
# @Author  : cap
# @FileName: myProcess.py
# @Software: PyCharm Community Edition

import multiprocessing as mp
import sys
import time


class MyProcess(mp.Process):
    def __init__(self, value):
        self.value = value
        super().__init__()

    def run(self):
        for i in range(5):
            print(time.ctime())
            time.sleep(2)


def main():
    p = MyProcess(2)
    p.start()
    p.join()


def func():
    time.sleep(2)
    print('child process')
    sys.exit(3)


def main1():
    p = mp.Process(target=func)
    p.start()

    print('is_alive:\t', p.is_alive())  # return whether process is alive
    # print('terminate:\t', p.terminate())
    print('authkey:\t', p.authkey)
    print('daemon:\t', p.daemon)
    print('ident:\t', p.ident)
    print('name:\t', p.name)
    print('pid:\t', p.pid)
    print('sentinel:\t', p.sentinel)
    p.join()

    print('exitcode:\t', p.exitcode)

if __name__ == '__main__':
    main()
