# @Time    : 2018/6/24 23:14
# @Author  : cap
# @FileName: myOpencvPCA.py
# @Software: PyCharm Community Edition
# @introduction: PCA
import numpy as np
import sklearn.decomposition as dc


def make_data():
    a = np.random.normal(size=250)
    b = np.random.normal(size=250)
    c = 2 * a + 3 * b
    d = 4 * a - b
    e = c + 2 * d
    x = np.c_[d, b, e, a, c]
    return x


def train_model(x):
    model = dc.PCA()
    model.fit(x)
    return model


def reduce_model(model, n_components, x):
    model.n_components = n_components
    x = model.fit_transform(x)
    return x


def main():
    x = make_data()
    model = train_model(x)
    # 重要性
    variances = model.explained_variance_
    print(variances)
    threshold = 0.8
    useful_indices = np.where(variances > threshold)
    x = reduce_model(model, len(useful_indices[0]), x)
    print(x)


if __name__ == '__main__':
    main()
