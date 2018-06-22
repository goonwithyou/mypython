# @Time    : 2018/6/22 21:28
# @Author  : cap
# @FileName: myOpencvFeature.py
# @Software: PyCharm Community Edition
# @introduction: 图像特征值
import cv2 as cv
import numpy as np
import matplotlib.pyplot as mp
import mpl_toolkits.axes_grid1 as mg


def read_image(filename):
    image = cv.imread(filename)
    return image


def show_image(title, image):
    cv.imshow(title, image)


def calc_features(image):
    # 用star提取颜色相关特征
    star = cv.xfeatures2d.StarDetector_create()
    keypoints = star.detect(image)
    # 用SIFT提取边缘等特征
    sift = cv.xfeatures2d.SIFT_create()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 对两个特征值进行整合
    keypoints, desc = sift.compute(gray, keypoints)
    return desc

def draw_desc(desc):
    ma = mp.matshow(desc, cmap='jet')
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('DESC', fontsize=20)
    mp.xlabel('Feature', fontsize=14)
    mp.ylabel('Sample', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator(8))
    ax.xaxis.set_minor_locator(mp.MultipleLocator())
    ax.yaxis.set_major_locator(mp.MultipleLocator(8))
    ax.yaxis.set_minor_locator(mp.MultipleLocator())
    mp.tick_params(which='both', top=True, right=True, labeltop=False, labelbottom=True, labelsize=10)
    dv = mg.make_axes_locatable(ax)
    ca = dv.append_axes('right', '3%', pad='3%')
    cb = mp.colorbar(ma, cax=ca)
    cb.set_label('DESC', fontsize=14)


def show_chart():
    mp.show()


def main():
    original = read_image('./data/penguin.jpg')
    desc = calc_features(original)
    print(desc.shape)
    draw_desc(desc)
    show_chart()
    # show_image('Original', original)
    # cv.waitKey()


if __name__ == '__main__':
    main()
