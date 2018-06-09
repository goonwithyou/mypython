import socket


st = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
st.connect(('localhost', 8000))
while True:
    sdata = input()
    if not sdata:
        break
    st.send(sdata.encode())
    data = st.recv(1024).decode()
    print(data)
st.close()
