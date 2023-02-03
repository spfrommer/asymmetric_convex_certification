import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR

import math

from convexrobust.data import datamodules
from convexrobust.model.abcrown_certifiable import ABCROWNCertifiable
from convexrobust.model.certificate import Norm
from convexrobust.model.modules import StandardMLP

from convexrobust.utils import dirs, file_utils
from convexrobust.utils import torch_utils as TU

# Hyperparameters are as in section G of https://arxiv.org/pdf/1805.12514.pdf

class ABCROWNMnist(ABCROWNCertifiable):
    def __init__(self, **kwargs):
        super().__init__(adv_norm='inf', adv_eps=np.linspace(0.001, 0.3, 20).tolist(), **kwargs)

        self.normalize = datamodules.get_normalize_layer('mnist_38')
        self.save_hyperparameters()
        self.model = StandardMLP(28 * 28, 2, [200])

    def forward(self, x):
        x = self.normalize(x)
        x = x.reshape(x.shape[0], -1)
        return self.model(x) + self.class_balance_prediction_shift().expand(x.shape[0], 2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = MultiStepLR(optimizer, milestones=range(30, self.epochs_n, 10), gamma=0.5)
        return [optimizer], [scheduler]


class ABCROWNFashionMnist(ABCROWNCertifiable):
    def __init__(self, **kwargs):
        super().__init__(adv_norm='inf', adv_eps=np.linspace(0.001, 0.1, 20).tolist(), **kwargs)

        self.normalize = datamodules.get_normalize_layer('fashion_mnist_shirts')
        self.save_hyperparameters()
        # self.model = CNN_Small(1)
        self.model = StandardMLP(28 * 28, 2, [200])

    def forward(self, x):
        # If do in one line, breaks...
        x = self.normalize(x)
        x = x.reshape(x.shape[0], -1)
        return self.model(x) + self.class_balance_prediction_shift().expand(x.shape[0], 2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = MultiStepLR(optimizer, milestones=range(30, self.epochs_n, 10), gamma=0.5)
        return [optimizer], [scheduler]


class ABCROWNMalimg(ABCROWNCertifiable):
    def __init__(self, **kwargs):
        super().__init__(adv_norm='inf', adv_eps=np.linspace(0.001, 0.1, 20).tolist(), **kwargs)

        self.normalize = datamodules.get_normalize_layer('malimg')
        self.save_hyperparameters()
        self.model = StandardMLP(512 * 512, 2, [200])

    def forward(self, x):
        x = self.normalize(x)
        x = x.reshape(x.shape[0], -1)
        return self.model(x) + self.class_balance_prediction_shift().expand(x.shape[0], 2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = MultiStepLR(optimizer, milestones=range(30, self.epochs_n, 10), gamma=0.5)
        return [optimizer], [scheduler]


class ABCROWNCifar(ABCROWNCertifiable):
    def __init__(self, **kwargs):
        super().__init__(adv_norm='inf', adv_eps=np.linspace(0.001, 2/255, 20).tolist(), **kwargs)

        self.normalize = datamodules.get_normalize_layer('cifar10_catsdogs')
        self.save_hyperparameters()
        self.model = StandardMLP(3 * 32 * 32, 2, [200])
        # self.model = CNN_Small(3)

    def forward(self, x):
        x = self.normalize(x)
        x = x.reshape(x.shape[0], -1)
        # Can't do broadcasting with ABCROWN
        return self.model(x) + self.class_balance_prediction_shift().expand(x.shape[0], 2)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.05)
        scheduler = MultiStepLR(optimizer, milestones=range(30, self.epochs_n, 10), gamma=0.5)
        return [optimizer], [scheduler]


def construct_cifar10_catsdogs():
    return ABCROWNCifar()

def construct_mnist_38():
    return ABCROWNMnist()

def construct_fashion_mnist_shirts():
    return ABCROWNFashionMnist()

def construct_malimg():
    return ABCROWNMalimg()

# https://github.com/locuslab/convex_adversarial/blob/master/examples/problems.py
# mnist_model() from above, this is CNN-A-Adv from the beta-crown paper
def CNN_A_Adv(in_channels, linear_size=7, out_classes=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*linear_size*linear_size,100),
        nn.ReLU(),
        nn.Linear(100, out_classes) # Change 10 -> 2 classes
    )

def CNN_Deep():
    return nn.Sequential(
        nn.Conv2d(3,8,4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8,8,3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8,8,3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8,8,4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 2)
    )

# A smaller CNN that is easier to verify than the above
def CNN_Small(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 16 * 8, 2)
    )

def abcrown_fashion_mnist_dataset(eps):
    return get_dataset('fashion_mnist_shirts-standard', eps)

def abcrown_mnist_38_dataset(eps):
    return get_dataset('mnist_38-standard', eps)

def abcrown_cifar10_catsdogs_dataset(eps):
    return get_dataset('cifar10_catsdogs-standard', eps)

def abcrown_malimg_dataset(eps):
    return get_dataset('malimg-standard', eps)

def get_dataset(name, eps):
    assert eps is not None

    data_path = dirs.out_path(name, 'data.pkl')
    data = file_utils.read_pickle(data_path)
    X, labels = data['signals'], data['targets']
    X, labels = torch.cat(X).to('cpu'), torch.cat(labels).to('cpu')
    # Only certify one class
    X, labels = X[labels == TU.CERT_CLASS], labels[labels == TU.CERT_CLASS]

    mean, std = torch.tensor(0.0), torch.tensor(1.0)  # Normalization happens in model

    data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))

    ret_eps = torch.reshape(eps / std, (1, -1, 1, 1))
    return X, labels, data_max, data_min, ret_eps

