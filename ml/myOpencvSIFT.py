# @Time    : 2018/6/22 21:22
# @Author  : cap
# @FileName: myOpencvSIFT.py
# @Software: PyCharm Community Edition
# @introduction: SIFT特征提取
import cv2 as cv

image = cv.imread('./data/table.jpg')
cv.imshow('Original', image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# 创建检测器
dectector = cv.xfeatures2d.SIFT_create()
keypoints = dectector.detect(gray)
cv.drawKeypoints(image, keypoints, image, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('Star', image)

cv.waitKey()
