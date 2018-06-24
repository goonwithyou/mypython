# @Time    : 2018/6/24 22:31
# @Author  : cap
# @FileName: myOpencvFace.py
# @Software: PyCharm Community Edition
# @introduction: 面部识别

# 视频捕捉
import cv2 as cv


def capture():
    """"
    视频捕捉
    """
    cap = cv.VideoCapture(0)
    while True:
        image = cap.read()[1]
        image = cv.resize(image, None, fx=0.75, fy=0.75, interpolation=cv.INTER_AREA)
        cv.imshow('video', image)
        if cv.waitKey(33) == 27:
            break
    cap.release()
    cv.destroyAllWindows()


def harr():
    """
    面部识别，通过opencv脚本捕捉面部眼睛鼻子
    :return:
    """
    # 创建面部识别脚本
    face_detector = cv.CascadeClassifier('./data/haar/face.xml')
    eye_detector = cv.CascadeClassifier('./data/haar/eye.xml')
    nose_detector = cv.CascadeClassifier('./data/haar/nose.xml')
    cap = cv.VideoCapture(0)
    while True:
        image = cap.read()[1]
        image = cv.resize(image, None, fx=0.75, fy=0.75, interpolation=cv.INTER_AREA)
        faces = face_detector.detectMultiScale(image, 1.3, 5)
        eye = eye_detector.detectMultiScale(image, 1.3, 5)
        nose = nose_detector.detectMultiScale(image, 1.3, 5)
        for l, t, w, h in faces:
            a, b = int(w / 2), int(h / 2)
            cv.ellipse(image, (l + a, t + b), (a, b), 0, 0, 360, (255, 0, 255), 2)
        for l, t, w, h in eye:
            a, b = int(w / 2), int(h / 2)
            cv.ellipse(image, (l + a, t + b), (a, b), 0, 0, 360, (255, 255, 0), 2)
        for l, t, w, h in nose:
            a, b = int(w / 2), int(h / 2)
            cv.ellipse(image, (l + a, t + b), (a, b), 0, 0, 360, (0, 255, 255), 2)

        cv.imshow('video', image)
        if cv.waitKey(33) == 27:
            break
    cap.release()
    cv.destroyAllWindows()


def pca():
    pass

def main():
    # 视频捕捉
    # capture()

    # 面部定位
    harr()

if __name__ == '__main__':
    main()
