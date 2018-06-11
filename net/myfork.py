import os


pid = os.fork()

if pid < 0:
    print('fail')
elif pid == 0:
    print('child process')
else:
    print('parent process')

print('end')
