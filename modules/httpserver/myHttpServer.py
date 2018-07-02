# @Time    : 2018/6/15 14:39
# @Author  : cap
# @FileName: myHttpServer.py
# @Software: PyCharm Community Edition

from socket import *
from threading import Thread


class MyTCPRequestHandler(object):
    def __init__(self, request):
        self.request = request

    def get_request_header(self):
        pass

    def status(self, status):
        if status == 404:
            responseHeaders = 'HTTP/1.1 404 not found\r\n'
            responseHeaders += '\r\n'
            responseHeaders += '==the page not found=='

    def handler(self):
        pass


class MyHandler(MyTCPRequestHandler):
    def handler(self):
        while True:
            data = self.request.recv(1024).decode()
            if data:
                print(data)
                self.request.send(b'i have receive')
            else:
                break

    def get(self):
        pass

    def post(self):
        pass

class MyHttpServer():
    def __init__(self, addr, handler):
        self.addr = addr
        self.handler = handler
        self.socketfd = MyHttpServer.create_socket(addr)

    @staticmethod
    def create_socket(addr):
        socketfd = socket(AF_INET, SOCK_STREAM)
        socketfd.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        socketfd.bind(addr)
        socketfd.listen()
        return socketfd

    def serve_forever(self):
        while True:
            connfd, connadd = self.socketfd.accept()
            my_tcp_request_handler = self.handler(connfd)
            # my_tcp_request_handler.get_request_header()
            myhandler = my_tcp_request_handler.handler
            ct = Thread(target=myhandler)
            ct.setDaemon(True)
            ct.start()


def main():
    ADDR = ('0.0.0.0', 9000)
    STATIC_DIR = './static'

    http_server = MyHttpServer(ADDR, MyHandler)
    http_server.serve_forever()


if __name__ == '__main__':
    main()
