# @Time    : 2018/6/9 16:24
# @Author  : cap
# @FileName: mylinear.py
# @Software: PyCharm Community Edition

import pickle
import numpy as np
import sklearn.linear_model as sl
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import matplotlib.patches as mc


def read_data(path):
    x, y = [], []
    data = np.loadtxt(path, delimiter=',')
    for raw in data:
        x.append(raw[0])
        y.append(raw[1])
    return np.array(x), np.array(y)


def train_model(train_x, train_y):
    model = sl.LinearRegression()
    model.fit(train_x, train_y)
    return model


def pred(model, x):
    return model.predict(x)


def eval_model(y, pred_y):
    mae = sm.mean_absolute_error(y, pred_y)
    mse = sm.mean_squared_error(y, pred_y)
    mda = sm.median_absolute_error(y, pred_y)
    evs = sm.explained_variance_score(y, pred_y)
    r2s = sm.r2_score(y, pred_y)
    print(round(mae, 2),
          round(mse, 2),
          round(mda, 2),
          round(evs, 2),
          round(r2s, 2))


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Linear Regression', fontsize=20)
    mp.xlabel('x', fontsize=14)
    mp.ylabel('y', fontsize=14)
    mp.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_train(x, y, pred_y):
    mp.plot(x, y, 's', label='Traing', color='red')
    sorted_indices = x.T[0].argsort()
    mp.plot(x.T[0][sorted_indices], pred_y[sorted_indices], '--', color='blue', label='Predict')
    mp.legend()


def draw_test(x, y, pred_y):
    mp.plot(x, y, 's', color='green', label='Testing')
    mp.plot(x, pred_y, 'o', color='orange', label='Test Predict')

    for x_i, pred_y_i, y_i in zip(x, pred_y, y):
        mp.gca().add_patch(mc.Arrow(x_i, pred_y_i, 0, y_i - pred_y_i,
                                    width=0.2, ec='none', fc='black'))
    mp.legend()


def show_chart():
    mp.show()


def main():
    x, y = read_data('./data/single.txt')
    train_size = int(len(x) * 0.8)
    train_x = x[:train_size].reshape(-1, 1)
    train_y = y[:train_size]

    model = train_model(train_x, train_y)
    pred_y = pred(model, train_x)
    eval_model(train_y, pred_y)

    test_x = x[train_size:].reshape(-1, 1)
    test_y = y[train_size:]
    pred_test_y = pred(model, test_x)
    eval_model(test_y, pred_test_y)

    #     save_model(model, './mymodel/my_single_model.mod')

    init_chart()
    draw_train(train_x, train_y, pred_y)
    draw_test(test_x, test_y, pred_test_y)
    show_chart()


if __name__ == '__main__':
    main()