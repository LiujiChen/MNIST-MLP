import numpy as np
import torch
import os

from MNIST import *
from torch.utils.data import DataLoader
from torch.optim import Adam

batch_size = 128
learning_rate = 0.001

model = MNISTMLPNet()
optimizer = Adam(model.parameters(), lr=learning_rate)
if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load('./model/model.pth'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pth'))


def train(epoch):
    print('load data......')
    train_set = MNISTSet()
    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    model.train()
    print('strat training......')
    for e in range(epoch):
        for i, (input, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(input)

            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()

            # print('epoch[{}/{}] batch[{}/{}]: loss={}, accuracy={}, output={}'.format(
            #     e+1, epoch, i+1, len(train_dataloader), loss.item(), accuracy(output, label).item(), output.max(dim=-1)[-1]
            # ))
            print('epoch[{}/{}] batch[{}/{}]: loss={}, accuracy={}'.format(
                e + 1, epoch, i + 1, len(train_dataloader), loss.item(), accuracy(output, label).item()
            ))
    torch.save(model, './model/model_{}.pth'.format(e))
    torch.save(optimizer, './model/optimizer_{}.pth'.format(e))
    # print('-----------------------------------------------------------------------------------')
    # print('test')
    # model.eval()
    # test(model)
    # model.train()

def test():
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
    train(2)
    test()
