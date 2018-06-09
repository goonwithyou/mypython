from socket import *


addr = ('localhost', 8000)
socketfd = socket(AF_INET, SOCK_DGRAM)

while True:
    try:
        msg = input('>>>')
    except (KeyboardInterrupt, EOFError):
        break
    if not msg:
        print('do not input null value!!')
        continue
    socketfd.sendto(msg.encode(), addr)
    if msg == 'bye':
        break
    data, addr = socketfd.recvfrom(1024)
    print(data.decode())

socketfd.close()
