# @Time    : 2018/6/16 14:04
# @Author  : cap
# @FileName: myVerifyCurve.py
# @Software: PyCharm Community Edition
# @introduction: 验证曲线
import csv
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
    encoders, x, y= [], [], []
    for row in range(len(data)):
        encoder = sp.LabelEncoder()
        if row < len(data) - 1:
            x.append(encoder.fit_transform(data[row]))
        else:
            y = encoder.fit_transform(data[row])
        encoders.append(encoder)
    x = np.array(x).T
    return x, y


def train_model_estimator(max_depth):
    model = se.RandomForestClassifier(
        max_depth=max_depth, random_state=7)
    return model


def eval_vc_estimator(model, x, y, n_estimators):
    train_score, test_score = sm.validation_curve(
        model, x, y, 'n_estimators',
        n_estimators, cv=5)
    print(train_score)
    print(test_score)
    return train_score, test_score


def train_model_max_depth(n_estimator):
    model = se.RandomForestClassifier(
        n_estimators=n_estimator, random_state=7)
    return model


def eval_vc_max_depth(model, x, y, max_depth):
    train_score, test_score = sm.validation_curve(
        model, x, y, 'max_depth',
        max_depth, cv=5)
    print(train_score)
    print(test_score)
    return train_score, test_score


def train_model(x, y):
    model = se.RandomForestClassifier(
        max_depth=8, n_estimators=200, random_state=7
    )
    model.fit(x, y)
    return model


def pred_model(model, x):
    return model.predict(x)


def init_estimator():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(121)
    mp.title('Training Curve on Extimator', fontsize=20)
    mp.xlabel('Number Of Estimators', fontsize=14)
    mp.ylabel('Accuracy', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator(20))
    ax.yaxis.set_major_locator(mp.MultipleLocator(0.1))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=20)
    mp.grid(linestyle=':')


def draw_estimator(n_estimators, train_score_estimator):
    mp.plot(n_estimators, train_score_estimator.mean(axis=1) * 100,
            'o-', c='dodgerblue', label='Train Score')
    mp.legend()


def init_max_depth():
    mp.subplot(122)
    mp.title('Training Curve on Max_depth', fontsize=20)
    mp.xlabel('Number Of Max_depth', fontsize=14)
    mp.ylabel('Accuracy', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_major_locator(mp.MultipleLocator(5))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=20)
    mp.grid(linestyle=':')


def draw_max_depth(n_max_depth, train_score_max_depth):
    mp.plot(n_max_depth, train_score_max_depth.mean(axis=1) * 100,
            'o-', c='orangered', label='Train Score')
    mp.legend()


def main():
    x, y = read_data('./data/car.txt')
    model_estimator = train_model_estimator(4)
    n_estimators = np.linspace(20, 200, 10).astype(int)
    train_score_estimators, test_socre_estimators = eval_vc_estimator(model_estimator, x, y, n_estimators)

    model_max_depth = train_model_max_depth(20)
    max_depths = np.linspace(1, 10, 10).astype(int)
    train_score_max_depth, test_score_max_depth = eval_vc_max_depth(model_max_depth, x, y, max_depths)

    init_estimator()
    draw_estimator(n_estimators, train_score_estimators)
    init_max_depth()
    draw_max_depth(max_depths, train_score_max_depth)
    mng = mp.get_current_fig_manager()
    mng.full_screen_toggle()
    mp.show()


if __name__ == '__main__':
    main()
