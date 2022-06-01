import struct
import numpy as np


from PIL import Image


def getImages(root):
    """
    to transform the storage format of images form IDX to np.array
    :param root: the location of images
    :return: images
    """
    with open(root, 'rb') as image_path:
        images_magic, images_num, rows, cols = struct.unpack('>IIII', image_path.read(16))
        images = np.fromfile(image_path, dtype=np.uint8).reshape(images_num, rows * cols)

    return images


def getLabels(root):
    """
    to get the label of image
    :param root: the location of images
    :return: labels
    """
    with open(root, 'rb') as label_path:
        labels_magic, labels_num = struct.unpack('>II', label_path.read(8))
        labels = np.fromfile(label_path, dtype=np.uint8)

    return labels


def getIndex(image_root, label_root):
    """
    to get the index of digital image
    :param image_root:
    :param label_root:
    :return:
    """
    labels = getLabels(label_root)

    index = [[], [], [], [], [], [], [], [], [], []]

    for i in range(len(labels)):
        index[labels[i]].append(i)

    return index


def showImage(image_array):
    """
    show image that in np.array format
    :param image_array:
    :return:
    """
    image = Image.fromarray(image_array.reshape(28, 28))
    image.show()

def accuracy(output, label):
    predict = output.max(dim=-1)[-1]
    acc = predict.eq(label).float().mean()
    return acc

if __name__ == '__main__':
    # # index = getIndex('data/unzip/train-images.idx3-ubyte', 'data/unzip/train-labels.idx1-ubyte')
    # # print(len(index[0]) + len(index[1]))
    # index = getIndex('data/unzip/t10k-images.idx3-ubyte', 'data/unzip/t10k-labels.idx1-ubyte')
    # print(len(index[0]) + len(index[1]))
    # images = getImages('data/unzip/train-images.idx3-ubyte')
    # print(images[0].shape)
    # showImage(images[0])
    labels = getLabels('data/unzip/t10k-labels.idx1-ubyte')
    print(labels[0])
