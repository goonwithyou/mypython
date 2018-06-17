# @Time    : 2018/6/16 17:21
# @Author  : cap
# @FileName: myDefineEncoder.py
# @Software: PyCharm Community Edition
# @introduction: 自定义编码器，分析收入
import numpy as np
import sklearn.naive_bayes as nb
import sklearn.ensemble as se
import sklearn.model_selection as sm
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
    # 低收入样本和高收入样本各取7500条
    num_less = 0
    num_more = 0
    max_each = 7500
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            # 不对带有？的数据就行采样，低收入高收入各取7500条
            if '?' not in line:
                line_data = line[:-1].split(', ')
                if line_data[-1] == '<=50K' and num_less < max_each:
                    data.append(line_data)
                    num_less += 1
                elif line_data[-1] == '>50K' and num_more < max_each:
                    data.append(line_data)
                    num_more += 1
                if num_less >= max_each and num_more >= max_each:
                    break
    data = np.array(data).T
    encoders, x = [], []
    for row in range(len(data)):
        # 如果第一个是数值类型，那么采用自定义的编码器
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
    # 使用朴素贝叶斯
    model = nb.GaussianNB()
    model.fit(x, y)
    return model


def pred_model(model, x):
    return model.predict(x)


# 确定预测精度
def eval_ac(y, pred_y):
    ac = (y == pred_y).sum() / pred_y.size
    print('Accuracy: {}%'.format(round(ac * 100, 2)))


def eval_cv(model, x, y):
    pc = sm.cross_val_score(model, x, y, cv=5, scoring='precision_weighted')
    rc = sm.cross_val_score(model, x, y, cv=5, scoring='recall_weighted')
    f1 = sm.cross_val_score(model, x, y, cv=5, scoring='f1_weighted')
    ac = sm.cross_val_score(model, x, y, cv=5, scoring='accuracy')
    print(round(pc.mean(), 2), round(rc.mean(), 2),
          round(f1.mean(), 2), round(ac.mean(), 2))


def make_data(encoders):
    data = [
        ['39', 'State-gov', '77516', 'Bachelors', '13', 'Never-married', 'Adm-clerical',
         'Not-in-family', 'White', 'Male', '2174', '0', '40', 'United-States']
    ]
    data = np.array(data).T
    x = []
    for row in range(len(data)):
        encoder = encoders[row]
        x.append(encoder.transform(data[row]))
    x = np.array(x).T
    return x


def main():
    encoders, x, y = read_data('./data/adult.txt')
    train_x, test_x, train_y, test_y = sm.train_test_split(
        x, y, test_size=0.25, random_state=5
    )
    model = train_model(train_x, train_y)
    eval_cv(model, x, y)
    pred_test_y = pred_model(model, test_x)
    eval_ac(test_y, pred_test_y)

    x = make_data(encoders)
    pred_y = pred_model(model, x)
    print(encoders[-1].inverse_transform(pred_y))


if __name__ == '__main__':
    main()
