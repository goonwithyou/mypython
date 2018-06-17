# @Time    : 2018/6/16 16:44
# @Author  : cap
# @FileName: myLearnCurve.py
# @Software: PyCharm Community Edition
# @introduction: 学习曲线

import numpy as np
import sklearn.naive_bayes as nb
import sklearn.ensemble as se
import sklearn.model_selection as sm
import sklearn.preprocessing as sp
import matplotlib.pyplot as mp


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            l = line[:-1].split(',')
            data.append(l)
    data = np.array(data).T
    encoders, x = [], []
    for row in range(len(data)):
        encoder = sp.LabelEncoder()
        if row < len(data) - 1:
            x.append(encoder.fit_transform(data[row]))
        else:
            y = encoder.fit_transform(data[row])
        encoders.append(encoder)
    x = np.array(x).T
    return x, y


def train_model():
    model = se.RandomForestClassifier(
        max_depth=8, n_estimators=200, random_state=7
    )
    return model


def eval_lc(model, x, y, train_sizes):
    train_sizes, train_scores, test_scores = sm.learning_curve(
        model, x, y, train_sizes=train_sizes, cv=5)
    print(train_scores)
    print(test_scores)
    return train_sizes, train_scores, test_scores


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Learning Curve', fontsize=20)
    mp.xlabel('Number of Training Samples', fontsize=14)
    mp.ylabel('Accuracy', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator(100))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(train_sizes, test_score):
    mp.plot(train_sizes, test_score.mean(axis=1) * 100,
            'o-', c='dodgerblue', label='Test Score')
    mp.legend()


def show_chart():
    mp.show()


def main():
    x, y = read_data('./data/car.txt')
    model = train_model()
    train_sizes = np.linspace(100, 1000, 10).astype(int)
    train_sizes, train_scores, test_scores = eval_lc(model, x, y, train_sizes)

    init_chart()
    draw_chart(train_sizes, test_scores)
    show_chart()


if __name__ == '__main__':
    main()
