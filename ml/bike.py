# @Time    : 2018/6/12 7:21
# @Author  : cap
# @FileName: bike.py
# @Software: PyCharm Community Edition

import csv
import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp


def read_data(filename, fb, fe):
    'fb:start column, fe:end column`'
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        x, y = [], []
        for row in reader:
            x.append(row[fb: fe])
            y.append(row[-1])
        fn = np.array(x[0])
        x = np.array(x[1:], dtype=float)
        y = np.array(x[1:], dtype=float)
        x, y = su.shuffle(x, y ,random_state=7)
    return fn, x, y


def train_model(x, y):
    model_dt = se.RandomForestRegressor(max_depth=10, n_estimators=1000, min_samples_split=2)
    model_dt.fit(x, y)
    return model_dt


def pred_model(model, x):
    return model.predict(x)


def eval_model(y, pred_y):
    mae = sm.mean_absolute_error(y, pred_y)
    mse = sm.mean_squared_error(y, pred_y)
    mde = sm.median_absolute_error(y, pred_y)
    evs = sm.explained_variance_score(y, pred_y)
    r2s = sm.r2_score(y, pred_y)
    print(mae, mse, mde, evs, r2s)


def init_model_day():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(211)
    mp.title('Random Forest Regression By Day', fontsize=16)
    mp.xlabel('Feature', fontsize=12)
    mp.ylabel('Importance', fontsize=12)
    mp.tick_params(which='both', top=True, right=True, labelright=True, lablesize=10)
    mp.grid(axis='y', linestyle=':')


def draw_model_day(fn_day, fi_day):
    fi_day = (fi_day * 100) / fi_day.max()
    sorted_indices = np.fliput(fi_day.argsort())
    pos = np.arange(sorted_indices.size)
    mp.bar(pos, fi_day[sorted_indices], align='center',
           facecolor='deepskyblue', edgecolor='steelblue',
           label='Day')
    mp.xticks(pos, fn_day[sorted_indices])
    mp.legend()


def draw_model_hour(fn_hour, fi_hour):
    fi_day = (fi_hour * 100) / fi_hour.max()
    sorted_indices = np.fliput(fi_day.argsort())
    pos = np.arange(sorted_indices.size)
    mp.bar(pos, fi_day[sorted_indices], align='center',
           facecolor='deepskyblue', edgecolor='steelblue',
           label='Decision Tree')
    mp.xticks(pos, fn_hour[sorted_indices])
    mp.legend()


def show_chart():
    mp.show()


def main():
    fn, x, y = read_data('/data/bike_day.csv')
    fn, x, y = read_data('/data/bike_hour.csv')
# 31 536

if __name__ == '__main__':
    main()