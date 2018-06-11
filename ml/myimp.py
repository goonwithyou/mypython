# @Time    : 2018/6/10 23:23
# @Author  : cap
# @FileName: imp.py.py
# @Software: PyCharm Community Edition

import numpy as np

import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.ensemble as se
import sklearn.metrics as sm

import matplotlib.pyplot as mp


def read_data():
    housing = sd.load_boston()
    fn = housing.feature_names
    x, y = su.shuffle(housing.data, housing.target, random_state=7)
    return fn, x, y


def train_model_dt(x, y):
    model_dt = st.DecisionTreeRegressor(max_depth=4)
    model_dt.fit(x, y)
    return model_dt


def train_model_ab(x, y):
    model_ab = se.AdaBoostRegressor(
        st.DecisionTreeRegressor(max_depth=4),
        n_estimators=400, random_state=7
    )
    model_ab.fit(x, y)
    return model_ab


def pred_model(model, x):
    return model.predict(x)


def eval_model(y, pred_y):
    mae = sm.mean_absolute_error(y, pred_y)
    mse = sm.mean_squared_error(y, pred_y)
    mda = sm.median_absolute_error(y, pred_y)
    evs = sm.explained_variance_score(y, pred_y)
    r2s = sm.r2_score(y, pred_y)
    result = [mae, mse, mda, evs, r2s]
    for i in map(round, result, [2]*5):
        print(i, end=' ')
    print()


def init_model_dt():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(211)
    mp.title('Decison Tree Regression', fontsize=20)
    mp.xlabel('Feature', fontsize=14)
    mp.ylabel('Importance', fontsize=14)
    mp.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)
    mp.grid(axis='y', linestyle=':')


def draw_model_dt(fn, fi_dt):
    fi_dt = (fi_dt * 100) / fi_dt.max()
    sorted_indices = np.flipud(fi_dt.argsort())
    pos = np.arange(sorted_indices.size)
    mp.bar(pos, fi_dt[sorted_indices], align='center',
           facecolor='deepskyblue', edgecolor='steelblue',
           label='Decision Tree')
    mp.xticks(pos, fn[sorted_indices])
    mp.legend()


def init_model_ab():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(212)
    mp.title('Ada Boost Regression', fontsize=20)
    mp.xlabel('Feature', fontsize=14)
    mp.ylabel('Importance', fontsize=14)
    mp.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)
    mp.grid(axis='y', linestyle=':')


def draw_model_ab(fn, fi_ab):
    fi_ab = (fi_ab * 100) / fi_ab.max()
    sorted_indices = np.flipud(fi_ab.argsort())
    pos = np.arange(sorted_indices.size)
    mp.bar(pos, fi_ab[sorted_indices], align='center',
           facecolor='lightcoral', edgecolor='indianred',
           label='Ada Boost Decision Tree')
    mp.xticks(pos, fn[sorted_indices])
    mp.legend()


def show_chart():
    mp.tight_layout()
    mp.show()


def main():
    fn, x, y = read_data()
    train_size = int(len(x) * 0.8)
    train_x = x[:train_size]
    train_y = y[:train_size]
    model1 = train_model_dt(train_x, train_y)
    model2 = train_model_ab(train_x, train_y)

    test_x = x[train_size:]
    test_y = y[train_size:]
    pred_test_y1 = pred_model(model1, test_x)
    pred_test_y2 = pred_model(model2, test_x)

    print('dt')
    eval_model(test_y, pred_test_y1)
    print('ab')
    eval_model(test_y, pred_test_y2)

    fi_dt = model1.feature_importances_
    fi_ab = model2.feature_importances_

    init_model_dt()
    draw_model_dt(fn, fi_dt)

    init_model_ab()
    draw_model_ab(fn, fi_ab)
    show_chart()


if __name__ == '__main__':
    main()
