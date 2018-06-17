# @Time    : 2018/6/12 12:42
# @Author  : cap
# @FileName: test.py.py
# @Software: PyCharm Community Edition
import numpy as np
import sklearn.preprocessing as sp

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            l = line[:-1].split(',')
            data.append(l)
    data = np.array(data).T
    print(data)
    encoders, x = [], []
    for row in range(len(data)):
        encoder = sp.LabelEncoder()
        if row < len(data) - 1:
            x.append(encoder.fit_transform(data[row]))
        else:
            y = encoder.fit_transform(data[row])
            encoder.inverse_transform()
        encoders.append(encoder)
    x = np.array(x).T
    return encoders, x, y

encoders, x, y = read_data('./data/car.txt')
print(encoders)
print(x)

