# @Time    : 2018/6/27 0:28
# @Author  : cap
# @FileName: myOpencvFaceRecog.py
# @Software: PyCharm Community Edition
# @introduction: 人脸识别
import os
import numpy as np
import cv2 as cv
import sklearn.preprocessing as sp


# 获取文件列表并得到{label:[path]}形式
# 编码器训练
# 人脸位置检测器
# 读取图像
# 灰度图处理
# 人脸区域检测
# 编码

def search_faces(directory):
    if not os.path.isdir(directory):
        raise IOError('the directory' + directory + 'is not exists!')
    faces = {}
    for curdir, subdir, files in os.walk(directory):
        for jpeg in (file for file in files if file.endswith('.jpg')):
            path = os.path.join(curdir, jpeg)
            label = path.split(os.path.sep)[-2]
            if label not in faces:
                faces[label] = []
            faces[label].append(path)
    return faces


# 加载脸部识别器
def load_detectors():
    face_detector = cv.CascadeClassifier('./data/haar/face.xml')
    return face_detector


# 由路径读取图片并return image
def read_image(filename):
    image = cv.imread(filename)
    return image


# 把图片变成gray
def bgr2gray(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


# 用脸部识别器，识别脸部范围，并返回脸部区域
def detect_regions(detector, image):
    regions = detector.detectMultiScale(image, 1.1, 2, minSize=(100, 100))
    return regions


# 定义编码器，用于将labels转成数字格式
def train_codec(labels):
    codec = sp.LabelEncoder()
    codec.fit(labels)
    return codec


# 用定义过的label编码器，获取label对应的code
def encode(codec, label):
    # transform接收的是列表，返回的是列表
    code = codec.transform([label])[0]
    return int(code)


# 将数字code转成label
def decode(codec, code):
    label = codec.inverse_transform(code)
    return label


# 显示图片
def show_image(title, image):
    cv.imshow(title, image)


# 定义图片关闭时间，毫秒，如果按esc则返回true
def wait_escape(delay=0):
    return cv.waitKey(delay) == 27


# 主要训练
def read_data(directory):
    # 获取文件列表和相应的label
    faces = search_faces(directory)
    # 训练label的编码器
    codec = train_codec(list(faces.keys()))
    # 加载脸部识别器
    face_detector = load_detectors()
    # x:截取脸部特征后的image，y:encode后的label,z:在原始图标出脸部区域
    x, y, z = [], [], []
    for label, filenames in faces.items():
        for filename in filenames:
            print(filename, '-->', label)
            # 读取图片
            original = read_image(filename)
            # 转成灰度图
            gray = bgr2gray(original)
            # 获取图片脸部范围,faces为一个二维坐标系
            faces = detect_regions(face_detector, gray)
            # 获取图片中的坐标，有可能一张图片出现两张脸
            for l, t, w, h in faces:
                # 截取脸部区域用于训练
                x.append(gray[t:t + h, l:l + w])
                # 保存每个截取图片的label并encode，用于训练
                y.append(encode(codec, label))
                #
                a, b = int(w / 2), int(h/2)
                # 在原图画椭圆（）
                cv.ellipse(original, (l + a, t + b), (a, b),
                           0, 0, 360, (255, 0, 255), 2)
                # 保留原图
                z.append(original)
    y = np.array(y)
    return codec, x, y, z


# 训练
def train_model(x, y):
    model = cv.face.LBPHFaceRecognizer_create()
    model.train(x, y)
    return model


# 预测
def pred_model(model, x):
    y = []
    # predict要一张张脸预测
    for face in x:
        y.append(model.predict(face)[0])
    return y


# 显示预测结果
def show_labels(codec, codes, pred_codes, images):
    escape = False
    while not escape:
        for code, pred_code, image in zip(codes, pred_codes, images):
            cv.putText(image, '{}{}{}'.format(
                decode(codec, code),
                '==' if code == pred_code else '!=',
                decode(codec, pred_code)
            ), (10, 60),cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255,255), 6)
            show_image('Recognizing Face...', image)
            if wait_escape(1000):
                escape = True
                break


def main():
    codec, train_x, train_y, train_z = read_data('./data/faces/training')
    _, test_x, test_y, test_z = read_data('./data/faces/testing')
    model = train_model(train_x, train_y)
    pred_test_y = pred_model(model, test_x)
    show_labels(codec, test_y, pred_test_y, test_z)


if __name__ == '__main__':
    main()
