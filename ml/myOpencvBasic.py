# @Time    : 2018/6/22 16:28
# @Author  : cap
# @FileName: myOpencvBasic.py
# @Software: PyCharm Community Edition
# @introduction: opencv_basic
import cv2 as cv
import numpy as np

image = cv.imread('./data/forest.jpg')
# 显示读取的图片
print(image.shape)
# print(image)
cv.imshow('Origimal', image)

# 对图片进行裁剪
h, w = image.shape[: 2]
l, t = int(w/4), int(h/4)
r, b = int(w*3/4), int(h*3/4)
cropped = image[t:b, l:r]
cv.imshow('cropped', cropped)

# 对三个颜色的通道进行拆分
blue = np.zeros_like(cropped)
blue[..., 0] = cropped[..., 0]
cv.imshow('blue', blue)

green = np.zeros_like(cropped)
green[..., 1] = cropped[..., 1]
cv.imshow('green', green)

red = np.zeros_like(cropped)
red[..., 2] = cropped[..., 2]
cv.imshow('red', red)

# 图片缩放
scaled = cv.resize(cropped, (w, h), interpolation=cv.INTER_LINEAR)
cv.imshow('scaled', scaled)

deformed = cv.resize(cropped, None, fx=2, fy=0.5, interpolation=cv.INTER_LINEAR)
cv.imshow('deformed', deformed)

# exit 按键盘任意键退出
cv.waitKey()
