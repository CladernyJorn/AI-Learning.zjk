import numpy as np
import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F  # useful stateless functions
import ssl
# 用于读取数据集路径防止错误（不知道为什么只能用没解压缩的数据集）
ssl._create_default_https_context = ssl._create_unverified_context


def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    # "flatten" the C * H * W values into a single vector per image
    return x.view(N, -1)


def check_accuracy_part34(loader, model):  # 不用调用
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' %
              (num_correct, num_samples, 100 * acc))


def train(model, optimizer, epochs=1):  # 默认softmax并在此处设置优化算法
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()

# 定义CNN的架构****************************************************************


class ConvNeuralNet(nn.Module):
    # Architecture I choose:
    # [conv1-relu-pool2-conv2-relu-pool2-conv3-relu-pool2]-> affine1->affine2 -> [softmax or SVM]
    # conv1:in_channel->channel_1
    # conv2:channel_1->channel_2
    # conv3:channel_2->channel_3
    # affine1: channel_3*32*32//64->hidden_size
    # affine2: hidden_size->num_classes
    def __init__(self, in_channel, channel_1, channel_2, channel_3, hidden_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, channel_1, (5, 5), padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(channel_1, channel_2, (5, 5), padding=2)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.conv3 = nn.Conv2d(channel_2, channel_3, (5, 5), padding=2)
        nn.init.kaiming_normal_(self.conv3.weight)

        self.fc1 = nn.Linear(channel_3*32*32//64, hidden_size)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        scores = None
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        scores = x
        return scores


#预处理并加载数据（使用GPU）###########################################################
USE_GPU = True
dtype = torch.float32  # We will be using float throughout this tutorial.

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss.
print_every = 100
print('using device:', device)

NUM_TRAIN = 49000

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

###########################################################################################
learning_rate = 1.2e-3
channel_1 = 20
channel_2 = 30
channel_3 = 50
hidden_size = 40
model = ConvNeuralNet(3, channel_1, channel_2, hidden_size, channel_3, 10)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train(model, optimizer, epochs=5)
