import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn
from utils import *


class MNISTSet(Dataset):
    def __init__(self,
                 root='data/unzip/',
                 image_file='train-images.idx3-ubyte',
                 labels_file='train-labels.idx1-ubyte',
                 transforms=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                 ]),
                 target_number=[0, 1]):
        self.root = root
        self.images_file = image_file
        self.labels_file = labels_file
        self.target_number = target_number
        self.transforms = transforms
        self.images = getImages(self.root + self.images_file)
        self.labels = getLabels(self.root + self.labels_file)
        self.images_index = getIndex(self.root + self.images_file, self.root + self.labels_file)
        self.target_index = self.getTargetIndex()

    def __getitem__(self, index):
        image = self.images[self.target_index[index]]
        image = self.transforms(Image.fromarray(image.reshape(28, 28)))
        label = self.labels[self.target_index[index]]
        label = label.astype(np.int64)
        return image, label

    def __len__(self):
        return len(self.target_index)

    def getTargetIndex(self):
        target_index = []
        for i in self.target_number:
            target_index += self.images_index[i]
        return target_index


class MNISTMLPNet(nn.Module):
    def __init__(self):
        super(MNISTMLPNet, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, input):
        x = input.view(input.size(0), 1 * 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return F.log_softmax(out, dim=-1)


if __name__ == '__main__':
    train_set = MNISTSet(image_file='t10k-images.idx3-ubyte', labels_file='t10k-labels.idx1-ubyte')
    print(len(train_set))

