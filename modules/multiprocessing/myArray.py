# @Time    : 2018/6/12 17:29
# @Author  : cap
# @FileName: myArray.py
# @Software: PyCharm Community Edition

from multiprocessing import Array
from multiprocessing import Process
import time


def func(arr):
    print('func arr:', list(arr))
    arr[0] = 100


def main():
    arr = Array('i', [1, 3, 5, 7, 9])
    # arr = Array('i', 5)
    p1 = Process(target=func, args=(arr, ))
    p1.start()
    p1.join()
    print('main arr', list(arr))


if __name__ == '__main__':
    main()
