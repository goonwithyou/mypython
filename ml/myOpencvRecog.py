# @Time    : 2018/6/24 9:51
# @Author  : cap
# @FileName: myOpencvRecog.py
# @Software: PyCharm Community Edition
# @introduction: 图像识别

import os
import warnings
import numpy as np
import cv2 as cv
import hmmlearn.hmm as hl


def search_file(directory):
    if not os.path.isdir(directory):
        raise IOError('this directory %s is not exist' % directory)
    objects = {}
    for curdir, subdir, files in os.walk(directory):
        for jpeg in (file for file in files if file.endswith('.jpg')):
            path = os.path.join(curdir, jpeg)
            label = os.path.basename(curdir)
            if label not in objects:
                objects[label] = []
            objects[label].append(path)
    return objects


def read_image(filename):
    return cv.imread(filename)


def resize_image(image, size):
    h, w = image.shape[:2]
    scale = size / min(h, w)
    image = cv.resize(image, None, fx=scale, fy=scale)
    return image


def calc_features(image):
    star = cv.xfeatures2d.StarDetector_create()
    keypoints = star.detect(image)

    sift = cv.xfeatures2d.SIFT_create()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    keypoints, desc = sift.compute(gray, keypoints)
    return desc


def read_data(directory):
    objects = search_file(directory)
    x, y, z = [], [], [] # desc, label , image
    for label, filenames in objects.items():
        z.append([])
        descs = np.array([])
        for filename in filenames:
            image = read_image(filename)
            z[-1].append(image)
            image = resize_image(image, 200)
            desc = calc_features(image)
            descs = desc if len(descs) == 0 else np.append(descs, desc, axis=0)
        x.append(descs)
        y.append(label)
    return x, y, z


def show_image(title, image):
    cv.imshow(title, image)


def train_models(x, y):
    models = {}
    for descs, label in zip(x, y):
        model = hl.GaussianHMM(n_components=4, covariance_type='diag', n_iter=1000)
        models[label] = model.fit(descs)
    return models


def pred_model(models, x):
    y = []
    for descs in x:
        best_score, best_label = None, None
        for label, model in models.items():
            score = model.score(descs)
            print(label, score)
            if best_score is None:
                best_score = score
            if best_label is None:
                best_label = label
            if best_score < score:
                best_score = score
                best_label = label
            print('best', best_label, best_score)
        y.append(best_label)
    return y


def show_labels(labels, pred_labels, images):
    i = 0
    for label, pred_label, row in zip(labels, pred_labels, images):
        for image in row:
            i += 1
            show_image('{}:{}{}{}'.format(i, label, '==' if label == pred_label else '!=', pred_label), image)
def main():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    # np.seterr(all='ignore')

    train_x, train_y, train_z = read_data('data\\objects\\training')
    test_x, test_y, test_z = read_data('data\\objects\\testing')
    models = train_models(train_x, train_y)
    pred_test_y = pred_model(models, test_x)
    show_labels(test_y, pred_test_y, test_z)
    cv.waitKey()


if __name__ == '__main__':
    main()
