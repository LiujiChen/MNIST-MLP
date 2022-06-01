import math
import os
import struct
from random import random

import numpy as np

from PIL import Image


def getImage():
    # 读取训练集图片
    with open('data/unzip/train-images.idx3-ubyte', 'rb') as image_path:
        images_magic, images_num, rows, cols = struct.unpack('>IIII', image_path.read(16))
        images = np.fromfile(image_path, dtype=np.uint8).reshape(images_num, rows * cols)
    return images


def getLabel():
    # 读取训练集图片标签
    with open('data/unzip/train-labels.idx1-ubyte', 'rb') as label_path:
        labels_magic, labels_num = struct.unpack('>II', label_path.read(8))
        labels = np.fromfile(label_path, dtype=np.uint8)
    return labels

def getIndex():
    images = getImage()
    labels = getLabel()

    index = [[], [], [], [], [], [], [], [], [], []]

    for i in range(len(images)):
        index[labels[i]].append(i)

    return index

if __name__ == '__main__':
    images = getImage()
    labels = getLabel()

    # print(len(images))
    # print(len(labels))
    index = getIndex()
    length = 0
    for i in index:
        length += len(i)
    print(length)

    # idx = 1
    # random_picture_number = math.floor(random() * len(index[idx]))
    # image = Image.fromarray(images[index[idx][random_picture_number]].reshape(28, 28))
    # label = labels[index[idx][random_picture_number]]
    #
    # image.show()
    # print(label)
    # print(random_picture_number)
