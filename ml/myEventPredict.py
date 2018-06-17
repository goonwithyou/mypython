# @Time    : 2018/6/17 11:22
# @Author  : cap
# @FileName: myEventPredict.py
# @Software: PyCharm Community Edition
# @introduction: 事件分类预测
import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.preprocessing as sp


class DigitEncoder():
    """自定义数值型编码器"""
    def fit_transform(self, y):
        return y.astype(int)

    def transform(self, y):
        return y.astype(int)

    def inverse_transform(self, y):
        return y.astype(str)


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line_data = line[:-1].split(',')
            data.append(line_data)
        data = np.delete(np.array(data).T, 1, 0)

        encoders, x = [], []
        for row in range(len(data)):
            if data[row, 0].isdigit():
                encoder = DigitEncoder()
            else:
                encoder = sp.LabelEncoder()
            if row < len(data) - 1:
                x.append(encoder.fit_transform(data[row]))
            else:
                y = encoder.fit_transform(data[row])
            encoders.append(encoder)
        x = np.array(x).T
        return encoders, x, y


def train_model(x, y):
    model = svm.SVC(kernel='rbf', class_weight='balanced')
    model.fit(x, y)
    return model


def pred_model(model, x):
    return model.predict(x)


# 确定预测精度
def eval_ac(y, pred_y):
    ac = (y == pred_y).sum() / pred_y.size
    print('Accuracy: {}%'.format(round(ac * 100, 2)))


def eval_cv(model, x, y):
    ac = ms.cross_val_score(model, x, y, cv=5, scoring='accuracy')
    print(round(ac.mean(), 2))


def make_data(encoders):
    data = [
        ['Tuesday', '12:30:00', '21', '23']
    ]
    data = np.array(data).T
    x = []

    for row in range(len(data)):
        encoder = encoders[row]
        x.append(encoder.transform(data[row]))
    x = np.array(x).T
    return x


def main():
    encoders, x, y = read_data('./data/event.txt')
    train_x, test_x, train_y, test_y = \
        ms.train_test_split(x, y, test_size=0.25, random_state=5)
    model = train_model(train_x, train_y)
    eval_cv(model, x, y)

    pred_y = pred_model(model, test_x)
    eval_ac(test_y, pred_y)

    new = make_data(encoders)
    pred_new = pred_model(model, new)
    print(encoders[-1].inverse_transform(pred_new))

if __name__ == '__main__':
    main()