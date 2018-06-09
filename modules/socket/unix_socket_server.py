import socket
import os


def create_socket(socket_file, backlog=None):
    socketfd = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    if os.path.exists(socket_file):
        os.remove(socket_file)
    socketfd.bind(socket_file)

    socketfd.listen(backlog)
    return socketfd


def read(conn, buffer_size=4096):
    data = None
    try:
        data = conn.recv(buffer_size).decode()
    except OSError as e:
        print('### Not receive msg, close fd:', conn.fileno())
        conn.close()
    return data


def send(conn, msg):
    try:
        conn.send(msg.encode())
    except OSError as e:
        print('### failure send!!')


def main():
    socket_file = './socket_file'
    socket_file_name = os.path.basename(socket_file)
    backlog = 10
    buffer_size = 1024
    msg = '***processing***'

    socketfd = create_socket(socket_file, backlog)

    conn, _ = socketfd.accept()
    while True:
        data = read(conn, buffer_size)
        if data:
            print('from %s Receive:%s' % (socket_file_name, data))
            send(conn, msg)
        else:
            break
    socketfd.close()


if __name__ == '__main__':
    main()
