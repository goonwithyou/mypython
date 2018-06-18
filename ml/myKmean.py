# @Time    : 2018/6/17 23:20
# @Author  : cap
# @FileName: myKmean.py
# @Software: PyCharm Community Edition
# @introduction: 聚类

import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp


def read_data(filename):
    x = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data = [float(substr) for substr in line.split(',')]
            x.append(data)
    return np.array(x)


def train_model(x):
    model = sc.KMeans(init='k-means++', n_clusters=4, n_init=10)
    model.fit(x)
    return model


def pred_model(model, x):
    return model.predict(x)


def init_chart():
    mp.gca().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('K-Means Cluster', fontsize=20)
    mp.xlabel('x', fontsize=16)
    mp.ylabel('y', fontsize=16)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)


def draw_grid(grid_x, grid_y):
    mp.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='brg')
    mp.xlim(grid_x[0].min(), grid_x[0].max())
    mp.ylim(grid_x[1].min(), grid_x[1].max())


def draw_data(x, y):
    mp.scatter(x[:, 0], x[:, 1], c=y, cmap='RdYlBu', s=80)


def draw_centers(centers):
    mp.scatter(centers[:, 0], centers[:, 1], marker='+', c='black',
               s=1000, linewidths=2)


def show_chart():
    mp.show()


def main():
    x = read_data('./data/multiple3.txt')
    l, r, h = x[:, 0].min() - 1, x[:, 0].max() + 1, 0.005
    b, t, v = x[:, 1].min() -1, x[:, 1].max() + 1, 0.005
    model = train_model(x)
    grid_x = np.meshgrid(np.arange(l, r, h),
                         np.arange(b, t, v))

    grid_y = pred_model(model,
                        np.c_[grid_x[0].ravel(),
                              grid_x[1].ravel()]).reshape(grid_x[0].shape)
    pred_y = pred_model(model, x)

    init_chart()
    draw_grid(grid_x, grid_y)
    draw_data(x, pred_y)
    draw_centers(model.cluster_centers_)
    show_chart()


if __name__ == '__main__':
    main()
