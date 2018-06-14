# @Time    : 2018/6/14 11:00
# @Author  : cap
# @FileName: myGILTest.py
# @Software: PyCharm Community Edition

# 计算密集
def count(x, y):
    for i in range(1000000):
        x += 1
        y += 1


# io密集
def write():
    with open('write.txt', 'w') as f:
        for i in range(10):
            f.write('asdfg')


def read():
    with open('write.txt', 'r') as f:
        for i in range(10):
            print(f.read(5))


def main():
    write()
    read()

if __name__ == '__main__':
    main()