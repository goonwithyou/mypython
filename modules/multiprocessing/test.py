# @Time    : 2018/6/12 9:54
# @Author  : cap
# @FileName: test.py.py
# @Software: PyCharm Community Edition
import time
import os
import multiprocessing as mp


def func():
    time.sleep(2)
    print('child process')

    def run():
        print('this is run')


def main():
    p = mp.Process(target=func)
    p.start()

    print('is_alive:\t', p.is_alive())  # return whether process is alive

    print('run:\t', p.run()) # Method to be run in sub-process; can be overridden in sub-class.

    print('terminate:\t', p.terminate())  # terminate process;sends SIGTERM signal or uses TerminateProcess()

    print('authkey:\t', p.authkey)

    print('daemon:\t', p.daemon)

    p.join()
    # print('exitcode:\t', p.exitcode)  # Return exit code of process or `None` if it has yet to stop
    #
    # print('ident:\t', p.ident)
    #
    # print('name:\t', p.name)
    #
    # print('pid:\t', p.pid)
    #
    # print('sentinel:\t', p.sentinel)

if __name__ == '__main__':
    main()
