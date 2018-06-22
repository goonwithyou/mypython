# @Time    : 2018/6/22 17:34
# @Author  : cap
# @FileName: myOpencvEqualize.py
# @Software: PyCharm Community Edition
# @introduction: 均衡化

import numpy as np
import cv2 as cv

image = cv.imread('./data/sunrise.jpg')
cv.imshow('Original', image)

# 转成灰度图
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# 对灰度图均衡化处理
equalized_gray = cv.equalizeHist(gray)
cv.imshow('Equalized Gray', equalized_gray)

# 转成YUV(亮度，色度，饱和度)
yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
# 对亮度进行均衡化处理，并转为BGR
yuv[..., 0] = cv.equalizeHist(yuv[..., 0])
equalized_color = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
cv.imshow('Equalized Color', equalized_color)

cv.waitKey()
