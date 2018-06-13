# @Time    : 2018/6/12 7:21
# @Author  : cap
# @FileName: bike.py
# @Software: PyCharm Community Edition

import csv
import numpy as np
import matplotlib.pyplot as mp
import platform


def make_data():
    x = np.array([
        [3, 1],
        [2, 5],
        [1, 8],
        [6, 4],
        [5, 2],
        [3, 5],
        [4, 7],
        [4, -1]
    ])
    y = np.array([0,1,1,0,0,1,1,0])
    return x, y


def pred_model(x):
    y = np.zeros(len(x), dtype=int)
    y[x[:, 0] < x[:, 1]] = 1
    return y


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Simple Classify', fontsize=16)
    mp.xlabel('x', fontsize=12)
    mp.ylabel('y', fontsize=12)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))
    mp.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)
    mp.grid(axis='y', linestyle=':')


def draw_data(x, y):
    mp.scatter(x[:, 0], x[:, 1], c=1-y, cmap='gray', s=80)


def draw_grid(grid_x, grid_y):
    mp.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='gray')
    mp.xlim(grid_x[0].min(), grid_x[0].max())
    mp.ylim(grid_x[1].min(), grid_x[1].max())


def show_chart():
    mp.show()


def main():
    x, y = make_data()
    l, r, h = x[:, 0].min() -1, x[:, 0].max() + 1, 0.005
    b, t, v = x[:, 1].min() -1, x[:, 1].max() + 1, 0.005
    grid_x = np.meshgrid(np.arange(l, r, h),
                         np.arange(b, t, v))
    grid_y = pred_model(
        np.c_[grid_x[0].ravel(), grid_x[1].ravel()]
    ).reshape(grid_x[0].shape)

    init_chart()
    draw_grid(grid_x, grid_y)
    draw_data(x, y)
    show_chart()

# 53 536

if __name__ == '__main__':
    main()