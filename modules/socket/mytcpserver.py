from socket import *


# create socket
socketfd = socket(AF_INET, SOCK_STREAM)
socketfd.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
# socketfd.bind(addr),addr=(host, port)
socketfd.bind(('localhost', 8000))
# socketfd.listen(n), set the max number of socket queue
socketfd.listen(5)
# wait for connect, return new socket and addr
# wait for receive
connfd, addr = socketfd.accept()

while True:
    data = connfd.recv(1024).decode('utf8')
    if not data:
        connfd.close()
        break
    print('From', addr, 'Receive', data)
    # send data to client
    connfd.send(b'I have received!')

socketfd.close()
