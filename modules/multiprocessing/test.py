# @Time    : 2018/6/12 9:54
# @Author  : cap
# @FileName: test.py.py
# @Software: PyCharm Community Edition
import threading


def func():
    while True:
        pass

func()
# th = []
# for i in range(2):
#     t = threading.Thread(target=func)
#     t.start()
#     th.append(t)
#     print(t.name)
#
# for i in th:
#     i.join()