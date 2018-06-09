from socket import *
import select


def main():
    HOST = '0.0.0.0'
    PORT = 8000
    dic_fd = {}

    # 创建socket服务
    socketfd = socket(AF_INET, SOCK_STREAM)
    socketfd.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    socketfd.bind((HOST, PORT))
    socketfd.listen(10)

    # 创建poll实例对象,并把sockedf注册到poll中，
    po = select.poll()
    po.register(socketfd, select.POLLIN | select.POLLERR)

    # 创建文件描述符字典
    dic_fd[socketfd.fileno()] = socketfd

    # print('in:', select.POLLIN) # 1
    # print('out:', select.POLLOUT) # 4
    # print('err:', select.POLLERR) # 8

    while True:
        events = po.poll()
        if events:
            for fn, event in events:
                if fn == socketfd.fileno():
                    connfd, addr = dic_fd[fn].accept()
                    po.register(connfd, select.POLLIN | select.POLLERR)
                    dic_fd[connfd.fileno()] = connfd
                elif event & select.POLLIN:
                    data = dic_fd[fn].recv(1024)
                    if not data:
                        po.unregister(fn)
                        dic_fd[fn].close()
                        print('has delete', dic_fd[fn])
                        del dic_fd[fn]
                    else:
                        print(dic_fd[fn].getpeername(), ':', data.decode())
                        # 把send的IO事件加入到wlist,等待下次循环执行
                        po.modify(fn, select.POLLIN | select.POLLERR |
                                  select.POLLOUT)
                elif event & select.POLLOUT:
                    dic_fd[fn].send(b'i have receive')
                    po.modify(fn, select.POLLIN | select.POLLERR)

                elif event & select.POLLERR:
                    po.unregister(fn)
                    del dic_fd[fn]


if __name__ == '__main__':
    main()
