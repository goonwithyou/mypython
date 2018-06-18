# @Time    : 2018/6/18 11:35
# @Author  : cap
# @FileName: mySpiral.py
# @Software: PyCharm Community Edition
# @introduction:
import numpy as np
import sklearn.cluster as sc
import sklearn.neighbors as nb
import matplotlib.pyplot as mp


def make_data(a_noise=0.05, n_samples=500):
    # 角
    t = (np.random.rand(n_samples, 1) * 2 + 1) * 2.5 * np.pi
    # 径
    x = 0.05 * t * np.cos(t)
    y = 0.05 * t * np.sin(t)
    n = a_noise * np.random.rand(n_samples, 2)
    return np.hstack((x, y)) + n


def train_model():
    model = sc.AgglomerativeClustering(linkage='average', n_clusters=3)
    return model


def train_model_10(x):
    model = sc.AgglomerativeClustering(linkage='average', n_clusters=3,
                                       connectivity=nb.kneighbors_graph(x, 10, include_self=False))
    return model


def pred_model(model, x):
    return model.fit_predict(x)


def init_chart():
    mp.gca().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(121)
    mp.title('Connectiviey: no', fontsize=20)
    mp.xlabel('x', fontsize=16)
    mp.ylabel('y', fontsize=16)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.1))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def init_chart_10():
    mp.subplot(122)
    mp.title('Connectiviey: 10', fontsize=20)
    mp.xlabel('x', fontsize=16)
    mp.ylabel('y', fontsize=16)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.1))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(x, y):
    mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg', s=80)


def show_chart():
    mp.show()


def main():
    x = make_data()
    model = train_model()
    model_10 = train_model_10(x)
    init_chart()
    draw_chart(x, pred_model(model, x))
    init_chart_10()
    draw_chart(x, pred_model(model_10, x))
    show_chart()
    # x = read_data('./data/multiple3.txt')
    # model = train_model()
    # pred_y = pred_model(model, x)
    #
    # init_chart()
    # draw_chart(x, pred_y)
    # show_chart()


if __name__ == '__main__':
    main()
