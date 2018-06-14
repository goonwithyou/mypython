# @Time    : 2018/6/14 23:06
# @Author  : cap
# @FileName: myTrainTest.py
# @Software: PyCharm Community Edition
import numpy as np
import matplotlib.pyplot as mp
import sklearn.naive_bayes as nb
import sklearn.model_selection as ms


def read_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def train_model(x, y):
    model = nb.GaussianNB()
    model.fit(x, y)
    return model


def eval_ac(y, pred_y):
    ac = (y == pred_y).sum() / pred_y.size
    print('Accuracy: {}%'.format(round(ac * 100, 2)))


def pred_model(model, x):
    return model.predict(x)


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Naive Bayes Classify', fontsize=16)
    mp.xlabel('x', fontsize=12)
    mp.ylabel('y', fontsize=12)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))
    mp.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)
    mp.grid(axis='y', linestyle=':')


def draw_train(train_x, train_y):
    mp.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap='RdYlBu', s=80)


def draw_test(test_x, test_y, pred_test_y):
    mp.scatter(test_x[:, 0], test_x[:, 1], c=test_y, marker='D', cmap='RdYlBu', s=80)
    mp.scatter(test_x[:, 0], test_x[:, 1], c=pred_test_y, marker='X', cmap='RdYlBu', s=80)


def draw_grid(grid_x, grid_y):
    mp.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='brg')
    mp.xlim(grid_x[0].min(), grid_x[0].max())
    mp.ylim(grid_x[1].min(), grid_x[1].max())


def show_chart():
    mp.show()


def main():
    x, y = read_data('./data/multiple1.txt')
    l, r, h = x[:, 0].min() -1, x[:, 0].max() + 1, 0.005
    b, t, v = x[:, 1].min() -1, x[:, 1].max() + 1, 0.005
    train_x, test_x, train_y, test_y = \
        ms.train_test_split(x, y, test_size=0.25, random_state=5)
    model = train_model(train_x, train_y)
    grid_x = np.meshgrid(np.arange(l, r, h),
                         np.arange(b, t, v))
    grid_y = pred_model(
        model,
        np.c_[grid_x[0].ravel(), grid_x[1].ravel()]
    ).reshape(grid_x[0].shape)

    pred_y = pred_model(model, test_x)
    eval_ac(test_y, pred_y)
    init_chart()
    draw_grid(grid_x, grid_y)
    draw_train(train_x, train_y)
    draw_test(test_x, test_y, pred_y)
    show_chart()


if __name__ == '__main__':
    main()
