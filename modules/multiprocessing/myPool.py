# @Time    : 2018/6/12 11:30
# @Author  : cap
# @FileName: myPool.py
# @Software: PyCharm Community Edition

from multiprocessing import Pool
from time import sleep


def worker(msg):
    sleep(1)
    print(msg)
    return msg


# map_asyc
def main():
    pool = Pool(processes=4)
    m = pool.map_async(worker, range(11))
    pool.close()
    pool.join()
    print(m.get())


# map
def main3():
    pool = Pool(processes=4)
    m = pool.map(worker, range(11))
    print(m)
    pool.close()
    pool.join()


# apply
def main2():
    pool = Pool(processes=4)

    for i in range(10):
        msg = 'hello' + str(i)
        pool.apply(func=worker, args=(msg,))

    pool.close()
    pool.join()


# apply_async
def main1():
    pool = Pool(processes=4)
    # save the return value of func
    results = []

    for i in range(10):
        msg = 'hello' + str(i)
        result = pool.apply_async(func=worker, args=(msg,))
        results.append(result)
    pool.close()
    pool.join()

    for o in results:
        print(o.get(), end=' ')


if __name__ == '__main__':
    main()
