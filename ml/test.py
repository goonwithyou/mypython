# @Time    : 2018/6/12 12:42
# @Author  : cap
# @FileName: test.py.py
# @Software: PyCharm Community Edition

with open('./data/adult.txt', 'rb') as f:
    l = f.read()
    print(len(l))