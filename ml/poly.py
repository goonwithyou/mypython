# @Time    : 2018/6/10 9:06
# @Author  : cap
# @FileName: poly.py
# @Software: PyCharm Community Edition

import numpy as np
import sklearn.pipeline as si
import sklearn.preprocessing as sp
import sklearn.linear_model as sl
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import matplotlib.patches as ma


def read_data(filename):
    x, y = [], []
    data = np.loadtxt(filename, delimiter=',')
    for d in data:
        x.append(d[0])
        y.append(d[1])
    return np.array(x), np.array(y)


def train_model(degree, x, y):
    model = si.make_pipeline(sp.PolynomialFeatures(degree),
                             sl.LinearRegression())
    model.fit(x, y)
    return model


def pred_model(model, x):
    return model.predict(x)


def eval_model(y, pred):
    mae = sm.mean_absolute_error(y, pred)
    mse = sm.mean_squared_error(y, pred)
    mda = sm.median_absolute_error(y, pred)
    evs = sm.explained_variance_score(y, pred)
    r2s = sm.r2_score(y, pred)
    result = [mae, mse, mda, evs, r2s]
    for i in map(round, result, [2]*5):
        print(i)


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Polynomial Regression', fontsize=20)
    mp.xlabel('x', fontsize=14)
    mp.ylabel('y', fontsize=14)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_train(train_x, train_y, pred_y):
    mp.plot(train_x, train_y, 's', c='green', label='Training')
    sort_indices = train_x.T[0].argsort()
    mp.plot(train_x.T[0][sort_indices], pred_y[sort_indices], '--', c='blue',
            label='Train Prediction')
    mp.legend()


def draw_test(test_x, test_y, pred_test_y):
    mp.plot(test_x, test_y, 's', c='yellow', label='Testing')
    mp.plot(test_x, pred_test_y, 'o', c='red', label='Test Predict')
    for x, pred_y, y in zip(test_x, pred_test_y, test_y):
        mp.gca().add_patch(ma.Arrow(x, pred_y, 0, y-pred_y,
                                    ec='none', fc='black',
                                    width=0.3))
    mp.legend()


def show_chart():
    mp.show()


def main():
    # read data
    x, y = read_data('./data/single.txt')

    # data preprocess
    train_size = int(x.size * 0.8)
    train_x = x[:train_size].reshape(-1, 1)
    train_y = y[:train_size]
    test_x = x[train_size:].reshape(-1, 1)
    test_y = y[train_size:]

    # train model and predict
    model = train_model(3, train_x, train_y)
    pred_y = pred_model(model, train_x)
    pred_test_y = pred_model(model, test_x)

    # error func
    print('test eval')
    eval_model(test_y, pred_test_y)

    # draw
    init_chart()
    draw_train(train_x, train_y, pred_y)
    draw_test(test_x, test_y, pred_test_y)
    show_chart()


if __name__ == '__main__':
    main()