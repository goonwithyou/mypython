# @Time    : 2018/6/18 11:23
# @Author  : cap
# @FileName: myAggloCluster.py
# @Software: PyCharm Community Edition
# @introduction: 凝聚层次聚类
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


def train_model():
    model = sc.AgglomerativeClustering(linkage='ward')
    return model


def pred_model(model, x):
    return model.fit_predict(x)


def init_chart():
    mp.gca().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Hierarchical Agglomerative Cluster', fontsize=20)
    mp.xlabel('x', fontsize=16)
    mp.ylabel('y', fontsize=16)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(x, y):
    mp.scatter(x[:, 0], x[:, 1], c=y, cmap='RdYlBu', s=80)


def show_chart():
    mp.show()


def main():
    x = read_data('./data/multiple3.txt')
    model = train_model()
    pred_y = pred_model(model, x)

    init_chart()
    draw_chart(x, pred_y)
    show_chart()


if __name__ == '__main__':
    main()
