# @Time    : 2018/6/15 17:44
# @Author  : cap
# @FileName: myGevent.py
# @Software: PyCharm Community Edition
import gevent
from gevent import monkey

monkey.patch_all()
from socket import *


def handle(c):
    while True:
        data = c.recv(1024).decode()
        if not data:
            break
        else:
            print(data)
            c.send(b'i have received')


def server():
    s = socket()
    s.bind('0.0.0.0', 9000)
    s.listen()

    while True:
        c, addr = s.accept()
        print('connect from', addr)
        gevent.spawn(handle, c)
