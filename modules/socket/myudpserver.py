from socket import *


# create socket
socketfd = socket(AF_INET, SOCK_DGRAM)
socketfd.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
# socketfd.bind(addr),addr=(host, port)
socketfd.bind(('localhost', 8000))

while True:
    data, addr = socketfd.recvfrom(1024)
    if data.decode() == 'bye':
        break
    print('From', addr, 'Receive', data.decode())
    # send data to client
    socketfd.sendto(b'I have received!', addr)

socketfd.close()
