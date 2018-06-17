# @Time    : 2018/6/17 10:55
# @Author  : cap
# @FileName: mySVMHyper.py
# @Software: PyCharm Community Edition
# @introduction: 自动选择超参数
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
    params = [
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
        {'kernel': ['poly'], 'C': [1], 'degree': [2, 3]},
        {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001]}
    ]
    model = ms.GridSearchCV(svm.SVC(probability=True), params, cv=5)
    model.fit(x, y)
    for i, param in enumerate(model.cv_results_['params']):
        print(param, model.cv_results_['mean_test_score'][i])
    print('best params', model.best_params_)
    return model


def pred_model(model, x):
    return model.predict(x)


def eval_cr(y, pred_y):
    cr = sm.classification_report(y, pred_y)
    print(cr)


def make_data():
    x = np.array([
        [2, 1.5],
        [8, 9],
        [4.8, 5.2],
        [4, 4],
        [3.5, 6.7],
        [7.6, 2],
        [5.4, 5.9],
    ])
    return x


def eval_cp(model, x):
    cp = model.predict_proba(x)
    return cp


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('SVM RBF Classifier', fontsize=16)
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


def draw_cp(cp_x, cp_y, cp):
    C0, C1 = cp_y == 0, cp_y == 1
    mp.scatter(cp_x[C0][:, 0], cp_x[C0][:, 1], marker='D', c='dodgerblue', s=80)
    mp.scatter(cp_x[C1][:, 0], cp_x[C1][:, 1], marker='D', c='deeppink', s=80)
    for i in range(len(cp[C0])):
        mp.annotate(
            '{}% {}%'.format(
                round(cp[C0][:, 0][i] * 100, 2),
                round(cp[C0][:, 1][i] * 100, 2),
            ),
            xy=(cp_x[C0][:, 0][i], cp_x[C0][:, 1][i]), # (x, y)坐标
            xytext=(12, -12), # 文字位置，右12点， 下12
            textcoords='offset points', # 文字偏移，相对偏移
            horizontalalignment='left', # 水平左对齐
            verticalalignment='top', # 垂直靠上
            fontsize=9,
            bbox={'boxstyle': 'round, pad=0.6',
                  'fc': 'deepskyblue', 'alpha':0.8} # 边框，前景色，厚度
        )
        for i in range(len(cp[C1])):
            mp.annotate(
                '{}% {}%'.format(
                    round(cp[C1][:, 0][i] * 100, 2),
                    round(cp[C1][:, 1][i] * 100, 2),
                ),
                xy=(cp_x[C1][:, 0][i], cp_x[C1][:, 1][i]),  # (x, y)坐标
                xytext=(12, -12),  # 文字位置，右12点， 下12
                textcoords='offset points',  # 文字偏移，相对偏移
                horizontalalignment='left',  # 水平左对齐
                verticalalignment='top',  # 垂直靠上
                fontsize=9,
                bbox={'boxstyle': 'round, pad=0.6',
                      'fc': 'deepskyblue', 'alpha': 0.8}  # 边框，前景色，厚度
            )

def show_chart():
    mp.show()


def main():
    x, y = read_data('./data/multiple2.txt')
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
    cp_x = make_data()
    cp_y = pred_model(model, cp_x)
    cp = eval_cp(model, cp_x)
    print(cp)
    init_chart()
    draw_grid(grid_x, grid_y)
    draw_data(x, y)
    draw_cp(cp_x, cp_y, cp)
    show_chart()


if __name__ == '__main__':
    main()
