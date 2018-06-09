from socket import *
from select import select


def main():
    HOST = '0.0.0.0'
    PORT = 8000

    rlist = []
    wlist = []
    xlist = []

    # 创建socket服务
    socketfd = socket(AF_INET, SOCK_STREAM)
    socketfd.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    socketfd.bind((HOST, PORT))
    socketfd.listen(10)

    # 将socket实例添加到rlist和xlist
    rlist.append(socketfd)
    xlist.append(socketfd)

    while True:
        rl, wl, xl = select(rlist, wlist, xlist)
        # 如果rl中有值则循环取出所有就绪IO事件，并执行相应的操作．
        if rl:
            for r in rl:
                # 如果r为socketfd则表明有新的客户端请求连接,
                # 如果不是则就是客户端socket实例
                if r is socketfd:
                    connfd, addr = r.accept()
                    # 把客户端socket添加到rlist准备监听
                    rlist.append(connfd)
                else:
                    data = r.recv(1024)
                    if not data:
                        # 如果没有接收到数据，则表明断开连接，
                        # 从rlist中移除该socket,并关闭该连接．
                        rlist.remove(r)
                        r.close()
                    else:
                        print(r.getpeername(), ':', data.decode())
                        # 把send的IO事件加入到wlist,等待下次循环执行
                        wlist.append(r)
                        # r.send(b'I hava receive')

        if wl:
            for w in wl:
                w.send(b'I have receive')
                # 发送IO执行完后，从wlist列表中移除．
                wlist.remove(w)

        if xl:
            for x in xl:
                if x is socketfd:
                    x.close()


if __name__ == '__main__':
    main()
