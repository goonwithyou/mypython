# @Time    : 2018/6/18 15:02
# @Author  : cap
# @FileName: myDbscan.py
# @Software: PyCharm Community Edition
# @introduction: dbscan

import numpy as np
import sklearn.cluster as sc
import sklearn.metrics as sm
import matplotlib.pyplot as mp


def read_data(filename):
    x = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data = [float(substr) for substr in line.split(',')]
            x.append(data)
    return np.array(x)


def train_model(x):
    epsilons = np.linspace(0.3, 1.2, 10)
    scores = []
    models = []
    for epsilon in epsilons:
        model = sc.DBSCAN(eps=epsilon, min_samples=5).fit(x)
        score = sm.silhouette_score(x, model.labels_, metric='euclidean',
                                    sample_size=len(x))
        scores.append(score)
        models.append(model)
    best_index = np.argmax(scores)
    best_epsilon = epsilons[best_index]
    best_score = scores[best_index]
    best_model = models[best_index]
    print(best_epsilon, best_score)
    return best_model


def pred_model(model, x):
    y = model.fit_predict(x)
    # 核心掩码
    core_mask = np.zeros(len(x), dtype=bool)
    core_mask[model.core_sample_indices_] = True
    # 带外掩码
    offset_mask = model.labels_ == -1
    # 边缘掩码
    periphery_mask = ~(core_mask | offset_mask)
    return y, core_mask, offset_mask, periphery_mask


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
    mp.grid(linestyle=':')



def draw_chart(x, y, core_mask, offset_mask, periphery_mask):
    labels = set(y)
    cs = mp.get_cmap('brg', len(labels))(range(len(labels)))
    mp.scatter(x[core_mask][:, 0], x[core_mask][:, 1],
               c=cs[y[core_mask]], s=80, label='core')
    mp.scatter(x[offset_mask][:, 0], x[offset_mask][:, 1],
               c=cs[y[offset_mask]], s=80,marker='x', label='offset')
    mp.scatter(x[periphery_mask][:, 0], x[periphery_mask][:, 1],
               edgecolor=cs[y[periphery_mask]],
               facecolor='none', s=80, label='periphery')
    mp.legend()



def show_chart():
    mp.show()


def main():
    x = read_data('./data/perf.txt')
    model = train_model(x)
    pred_y, core_mask, offset_mask, periphery_mask = pred_model(model, x)

    init_chart()
    draw_chart(x, pred_y, core_mask, offset_mask, periphery_mask)
    show_chart()


if __name__ == '__main__':
    main()
