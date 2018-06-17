# @Time    : 2018/6/16 10:20
# @Author  : cap
# @FileName: myReport.py.py
# @Software: PyCharm Community Edition
# @introduction: 性能报告

import numpy as np
import matplotlib.pyplot as mp
import sklearn.naive_bayes as nb
import sklearn.metrics as sm
import sklearn.model_selection as ms
import mpl_toolkits.axes_grid1 as mg


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


def eval_cm(y, pred_y):
    cm = sm.confusion_matrix(y, pred_y)
    # print(cm)
    return cm


def eval_cr(y, pred_y):
    cr = sm.classification_report(y, pred_y)
    print(cr)


def eval_cv(model, x, y):
    pc = ms.cross_val_score(model, x, y, cv=10, scoring='precision_weighted')
    rc = ms.cross_val_score(model, x, y, cv=10, scoring='recall_weighted')
    f1 = ms.cross_val_score(model, x, y, cv=10, scoring='f1_weighted')
    ac = ms.cross_val_score(model, x, y, cv=10, scoring='accuracy')
    print(round(pc.mean(), 2), round(rc.mean(), 2),
          round(f1.mean(), 2), round(ac.mean(), 2))


def pred_model(model, x):
    return model.predict(x)


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(121)
    mp.title('Confusion Matrix', fontsize=16)
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


def init_cm():
    mp.subplot(122)
    mp.title('Confusion Matrix', fontsize=16)
    mp.xlabel('Predicted Class', fontsize=12)
    mp.ylabel('True Class', fontsize=12)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    mp.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)


def draw_cm(cm):
    im = mp.imshow(cm, interpolation='nearest', cmap='jet')
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            mp.text(col, row, str(cm[row, col]), color='orangered',
                    fontsize=14, fontstyle='italic',
                    ha='center', va='center')
    dv = mg.make_axes_locatable(mp.gca())
    ca = dv.append_axes('right', '6%', pad='5%')
    cb = mp.colorbar(im, cax=ca)
    cb.set_label('Number Of Samples')


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
    eval_cv(model, test_x, pred_y)
    cm = eval_cm(test_y, pred_y)
    eval_cr(test_y, pred_y)
    # init_chart()
    # draw_grid(grid_x, grid_y)
    # draw_train(train_x, train_y)
    # draw_test(test_x, test_y, pred_y)
    #
    # init_cm()
    # draw_cm(cm)
    # show_chart()


if __name__ == '__main__':
    main()
