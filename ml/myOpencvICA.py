# @Time    : 2018/6/26 23:08
# @Author  : cap
# @FileName: myOpencvICA.py
# @Software: PyCharm Community Edition
# @introduction: 独立成分分析

import numpy as np
import sklearn.decomposition as dc
import matplotlib.pyplot as mp


def read_data(filename):
    x = np.loadtxt(filename)
    return x


def ica(x):
    model = dc.FastICA(n_components=x.shape[1])
    x = model.fit_transform(x)
    return x


def init_original():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(211)
    mp.title('Original', fontsize=18)
    mp.xlabel('Time', fontsize=14)
    mp.ylabel('Signal', fontsize=14)
    mp.tick_params(which='both', right=True, labelright=True, top=True, labelsize=10)
    mp.grid(linestyle=':')


def init_ica():
    mp.subplot(212)
    mp.title('ICA', fontsize=18)
    mp.xlabel('Time', fontsize=14)
    mp.ylabel('Signal', fontsize=14)
    mp.tick_params(which='both', right=True, labelright=True, top=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(x):
    x = x.T
    for i, compoent in enumerate(x):
        mp.plot(compoent, label='Componet %d' % (i + 1))
    # mp.plot(x.sum(axis=0), label='Mixture')
    mp.legend()


def show_chart():
    mp.tight_layout()
    mp.show()


def main():
    x = read_data('./data/signals.txt')
    ica_x = ica(x)
    init_original()
    draw_chart(x)
    init_ica()
    draw_chart(ica_x)
    show_chart()


if __name__ == '__main__':
    main()
