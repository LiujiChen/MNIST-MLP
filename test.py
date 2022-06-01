import numpy as np
import torch
from torch.utils.data import DataLoader

from MNIST import *
from utils import accuracy


def test():
    model = torch.load('./model/model_9.pth')
    test_set = MNISTSet(image_file='t10k-images.idx3-ubyte', labels_file='t10k-labels.idx1-ubyte')
    test_dataloader = DataLoader(dataset=test_set, batch_size=128)
    # model = MNISTMLPNet()
    # model.load_state_dict(torch.load('./model/model_1.pkl'))
    model.eval()

    loss = []
    acc = []
    for idx, (input, label) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(input)
            batch_loss = F.nll_loss(output, label)
            batch_acc = accuracy(output, label)
            loss.append(batch_loss.item())
            acc.append(batch_acc.item())
            # print('batch[{}/{}]: loss={}, accuracy={}, output={}'.format(
            #     idx+1, len(test_dataloader), batch_loss, batch_acc, output.max(dim=-1)[-1]
            # ))
            print('batch[{}/{}]: loss={}, accuracy={}'.format(
                idx + 1, len(test_dataloader), batch_loss, batch_acc
            ))

    print('loss={}, acc={}'.format(np.mean(loss), np.mean(acc)))

if __name__ == '__main__':
    test()