# @Time    : 2018/6/17 10:02
# @Author  : cap
# @FileName: mySVMImbalance.py
# @Software: PyCharm Community Edition
# @introduction: 带有均衡调节的分类器
import numpy as np
import matplotlib.pyplot as mp
import sklearn.svm as svm
import sklearn.metrics as sm
import sklearn.model_selection as ms


def read_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def train_model(x, y):
    model = svm.SVC(kernel='linear', class_weight='balanced')
    model.fit(x, y)
    return model


def pred_model(model, x):
    return model.predict(x)


def eval_cr(y, pred_y):
    cr = sm.classification_report(y, pred_y)
    print(cr)


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('SVM Linear Classifier', fontsize=16)
    mp.xlabel('x', fontsize=12)
    mp.ylabel('y', fontsize=12)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))
    mp.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)


def draw_grid(grid_x, grid_y):
    mp.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='gray')
    mp.xlim(grid_x[0].min(), grid_x[0].max())
    mp.ylim(grid_x[1].min(), grid_x[1].max())


def draw_data(x, y):
    C0, C1 = y == 0, y == 1
    mp.scatter(x[C0][:, 0], x[C0][:, 1], c='orangered', s=80)
    mp.scatter(x[C1][:, 0], x[C1][:, 1], c='limegreen', s=80)


def show_chart():
    mp.show()


def main():
    x, y = read_data('./data/imbalance.txt')
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
    eval_cr(test_y, pred_y)
    init_chart()
    draw_grid(grid_x, grid_y)
    draw_data(x, y)
    show_chart()


if __name__ == '__main__':
    main()
