# @Time    : 2018/6/22 17:03
# @Author  : cap
# @FileName: myOpencvEdge.py
# @Software: PyCharm Community Edition
# @introduction: 边缘检测
import cv2 as cv

image = cv.imread('./data/chair.jpg', cv.IMREAD_GRAYSCALE)
print(image.shape)
print(image.dtype)
cv.imshow('chair', image)

# 1 索贝尔边缘检测
# 检测水平梯度变化, ksize:卷积框
hor = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
cv.imshow('Hor', hor)
# 检测垂直梯度变化
ver = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)
cv.imshow('Ver', ver)
# 检测两个方向
hor_ver = cv.Sobel(image, cv.CV_64F, 1, 1, ksize=5)
cv.imshow('Hor_Ver', hor_ver)

# 2 拉普拉斯边缘检测
lap = cv.Laplacian(image, cv.CV_64F)
cv.imshow('Lap', lap)

# 3 Canny 50:
canny = cv.Canny(image, 50, 240)
cv.imshow('Canny', canny)

cv.waitKey()