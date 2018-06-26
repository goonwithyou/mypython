# @Time    : 2018/6/25 23:51
# @Author  : cap
# @FileName: myOpencvKPCA.py
# @Software: PyCharm Community Edition
# @introduction: 核主成分分析

import numpy as np
import sklearn.datasets as sd
import sklearn.decomposition as dc
import matplotlib.pyplot as mp


def make_data():
    np.random.seed(7)
    x, y = sd.make_circles(n_samples=500, factor=0.2, noise=0.04)
    return x, y


def pca(x):
    model = dc.PCA()
    x = model.fit_transform(x)
    return x


def kpca(x):
    model = dc.KernelPCA(kernel='rbf', fit_inverse_transform=True, gamma=10)
    x = model.fit_transform(x)
    return model, x


def ikpca(model, x):
    x = model.inverse_transform(x)
    return x


def init_original():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(221)
    mp.title('Original Samples', fontsize=18)
    mp.xlabel('x', fontsize=12)
    mp.ylabel('y', fontsize=12)
    mp.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def init_pca():
    mp.subplot(222)
    mp.title('PCA Samples', fontsize=18)
    mp.xlabel('x', fontsize=12)
    mp.ylabel('y', fontsize=12)
    mp.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def init_kpca():
    mp.subplot(223)
    mp.title('KPCA Samples', fontsize=18)
    mp.xlabel('x', fontsize=12)
    mp.ylabel('y', fontsize=12)
    mp.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def init_ikpca():
    mp.subplot(224)
    mp.title('IKPCA Samples', fontsize=18)
    mp.xlabel('x', fontsize=12)
    mp.ylabel('y', fontsize=12)
    mp.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(x, y):
    mp.scatter(x[y==0][:, 0], x[y==0][:, 1], c='dodgerblue', alpha=0.5, s=80, label='Class 0')
    mp.scatter(x[y==1][:, 0], x[y==1][:, 1], c='Orangered', alpha=0.5, s=80, label='Class 1')
    mp.legend()


def show_chart():
    mp.tight_layout()
    mp.show()


def main():
    x, y = make_data()
    pca_x = pca(x)
    model, kpca_x = kpca(x)
    i_x = ikpca(model, kpca_x)

    init_original()
    draw_chart(x, y)

    init_pca()
    draw_chart(pca_x, y)

    init_kpca()
    draw_chart(kpca_x, y)

    init_ikpca()
    draw_chart(i_x, y)
    show_chart()


if __name__ == '__main__':
    main()
