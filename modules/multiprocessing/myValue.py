# @Time    : 2018/6/12 17:15
# @Author  : cap
# @FileName: myValue.py
# @Software: PyCharm Community Edition

from multiprocessing import Value
from multiprocessing import Process
import random
import time


def deposite(money):
    for i in range(100):
        time.sleep(0.03)
        money.value += random.randint(1, 200)


def with_draw(money):
    for i in range(100):
        time.sleep(0.04)
        money.value -= random.randint(1, 200)


def main():
    money = Value('i', 2000)
    p1 = Process(target=deposite, args=(money,))
    p2 = Process(target=with_draw, args=(money,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print(money.value)


if __name__ == '__main__':
    main()
