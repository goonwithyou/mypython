# @Time    : 2018/6/22 18:08
# @Author  : cap
# @FileName: myOpencvStar.py
# @Software: PyCharm Community Edition
# @introduction: star特征检测
import cv2 as cv

image = cv.imread('./data/table.jpg')
cv.imshow('Original', image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# 创建检测器
dectector = cv.xfeatures2d.StarDetector_create()
keypoints = dectector.detect(gray)
cv.drawKeypoints(image, keypoints, image, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('Star', image)

cv.waitKey()

