from socket import *


socketfd = socket(AF_INET, SOCK_STREAM)
socketfd.connect(('localhost', 8000))
while True:
    try:
        msg = input('>>>')
    except (KeyboardInterrupt, EOFError):
        break
    if not msg:
        print('do not input null value!!')
        continue
    socketfd.send(msg.encode())
    data = socketfd.recv(1024).decode()
    if not data:
        break
    print(data)

socketfd.close()
