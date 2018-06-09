import datetime as dt
import numpy as np


def py_sum(n):
    L1 = [i ** 2 for i in range(n)]
    L2 = [i ** 3 for i in range(n)]
    L3 = []
    for i in range(n):
        L3.append(L1[i] + L2[i])
    return L3


def np_sum(n):
    return np.arange(n) ** 2 + np.arange(n) ** 3


def main():
    start1 = dt.datetime.now()
    py_sum(100000)
    end1 = dt.datetime.now()
    print('py-->', (end1 - start1).microseconds)

    start2 = dt.datetime.now()
    np_sum(100000)
    end2 = dt.datetime.now()
    print('np-->', (end2 - start2).microseconds)


main()
