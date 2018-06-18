# @Time    : 2018/6/18 17:28
# @Author  : cap
# @FileName: mySales.py
# @Software: PyCharm Community Edition
# @introduction: 消费习惯预测

import numpy as np
import csv
import sklearn.cluster as sc
import matplotlib.pyplot as mp


def read_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        x = []
        for row in reader:
            x.append(row[2:])
    fn = np.array(x[0])
    x = np.array(x[1:], dtype=float)
    return fn, x


def train_model(x):
    bw = sc.estimate_bandwidth(x, quantile=0.8, n_samples=len(x))
    model = sc.MeanShift(bandwidth=bw, bin_seeding=True)
    model.fit(x)
    return model


def pred_model(model, x):
    y = model.predict(x)
    return y

def init_chart():
    mp.gca().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Milk And Grocery Sales Volume Cluster', fontsize=20)
    mp.xlabel('Milk', fontsize=16)
    mp.ylabel('Grocery', fontsize=16)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator(10000))
    ax.xaxis.set_minor_locator(mp.MultipleLocator(5000))
    ax.yaxis.set_major_locator(mp.MultipleLocator(10000))
    ax.yaxis.set_minor_locator(mp.MultipleLocator(5000))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(x, y):
    mp.scatter(x[:, 0], x[:, 1],
               c=y, cmap='brg', s=80, label='core', alpha=0.5)


def show_chart():
    mp.show()


def main():
    fn, x = read_data('./data/sales.csv')
    model = train_model(x)
    pred_y = pred_model(model, x)

    init_chart()
    draw_chart(x, pred_y)
    show_chart()
    for cls in np.unique(pred_y):
        cls_idx = np.where(pred_y==cls)
        cls_sam = x[cls_idx]
        for row, cls in zip(cls_sam, pred_y[cls_idx]):
            for feature in row:
                print('{:6.0f}'.format(feature), end='\t')
            print(cls)

if __name__ == '__main__':
    main()
