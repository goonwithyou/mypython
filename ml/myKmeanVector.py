# @Time    : 2018/6/18 9:00
# @Author  : cap
# @FileName: myKmeanVector.py
# @Software: PyCharm Community Edition
# @introduction: 图像矢量化处理

import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp
import scipy.misc as sm


def train_model(n_clusters, x):
    model = sc.KMeans(n_clusters=n_clusters, n_init=10, random_state=5)
    model.fit(x)
    return model


def load_image(filename):
    return sm.imread(filename, True).astype(np.uint8)


# 压缩图像bpp为要压缩的位数
def compress_image(image, bpp):
    n_cluster = np.power(2, bpp)
    x = image.reshape((-1, 1))
    model = train_model(n_cluster, x)
    y = model.labels_
    centers = model.cluster_centers_.squeeze()
    z = centers[y]
    return z.reshape(image.shape)


def init_chart():
    mp.gca().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Compress Image', fontsize=20)
    mp.axis('off')


def draw_chart(image):
    mp.imshow(image, cmap='gray')


def show_chart():
    mp.show()


def main():
    image = load_image('./data/flower.jpg')
    compressed_image = compress_image(image, 1)
    init_chart()
    draw_chart(compressed_image)
    show_chart()


if __name__ == '__main__':
    main()
