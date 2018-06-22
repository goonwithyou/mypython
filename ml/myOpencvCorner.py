# @Time    : 2018/6/22 17:55
# @Author  : cap
# @FileName: myOpencvCorner.py
# @Software: PyCharm Community Edition
# @introduction: 棱角检测

import cv2 as cv

image = cv.imread('./data/box.png')
cv.imshow('Original', image)

# 转成灰度图
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

corner = cv.cornerHarris(gray, 10, 5, 0.04)
# corner的值较小，需要放大处理才能显示，这里通过掩码的形式，对原图进行处理
corner = cv.dilate(corner, None)
threshold = corner.max() * 0.01
corner_mask = corner > threshold
image[corner_mask] = [0, 0, 255]
cv.imshow('Corner', image)

cv.waitKey()