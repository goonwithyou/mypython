import socket


socket_file = './socket_file'
socketfd = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
socketfd.connect(socket_file)

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
print('data')

socketfd.close()
