# @Time    : 2018/6/16 10:35
# @Author  : cap
# @FileName: myCarReport.py
# @Software: PyCharm Community Edition
# @introduction: 汽车质量评估

import csv
import numpy as np
import sklearn.naive_bayes as nb
import sklearn.ensemble as se
import sklearn.model_selection as sm
import sklearn.preprocessing as sp


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            l = line[:-1].split(',')
            data.append(l)
    data = np.array(data).T
    encoders, x = [], []
    for row in range(len(data)):
        encoder = sp.LabelEncoder()
        if row < len(data) - 1:
            x.append(encoder.fit_transform(data[row]))
        else:
            y = encoder.fit_transform(data[row])
        encoders.append(encoder)
    x = np.array(x).T
    return encoders, x, y


def train_model(x, y):
    model = se.RandomForestClassifier(
        max_depth=8, n_estimators=200, random_state=7
    )
    model.fit(x, y)
    return model


def pred_model(model, x):
    return model.predict(x)


def eval_ac(y, pred_y):
    ac = (y == pred_y).sum() / pred_y.size
    print('Accuracy: {}%'.format(round(ac * 100, 2)))


def eval_cv(model, x, y):
    pc = sm.cross_val_score(model, x, y, cv=2, scoring='precision_weighted')
    rc = sm.cross_val_score(model, x, y, cv=3, scoring='recall_weighted')
    f1 = sm.cross_val_score(model, x, y, cv=2, scoring='f1_weighted')
    ac = sm.cross_val_score(model, x, y, cv=3, scoring='accuracy')
    print(round(pc.mean(), 2), round(rc.mean(), 2),
          round(f1.mean(), 2), round(ac.mean(), 2))


def make_data(encoders):
    data = [
        ['high', 'med', '5more', '4', 'big', 'low', 'unacc'],
        ['high', 'high', '4', '4', 'med', 'med', 'acc'],
        ['low', 'low', '2', '4', 'small', 'high', 'good'],
        ['low', 'med', '3', '4', 'med', 'high', 'vgood']
    ]
    data = np.array(data).T
    x = []
    for row in range(len(data)):
        encoder = encoders[row]
        if row < len(data) - 1:
            x.append(encoder.transform(data[row]))
        else:
            y = encoder.transform(data[row])
    x = np.array(x).T
    return x, y


def main():
    encoders, train_x, train_y = read_data('./data/car.txt')
    model = train_model(train_x, train_y)
    eval_cv(model, train_x, train_y)
    test_x, test_y = make_data(encoders)
    pred_test_y = pred_model(model, test_x)
    eval_ac(test_y, pred_test_y)
    print(encoders[-1].inverse_transform(test_y))
    print(encoders[-1].inverse_transform(pred_test_y))


if __name__ == '__main__':
    main()
