# @Time    : 2018/6/10 21:14
# @Author  : cap
# @FileName: housing.py
# @Software: PyCharm Community Edition

import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.ensemble as se
import sklearn.metrics as sm


def read_data():
    housing = sd.load_boston()
    x, y = su.shuffle(housing.data, housing.target, random_state=7)
    return x, y


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


def main():
    x, y = read_data()
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


if __name__ == '__main__':
    main()
