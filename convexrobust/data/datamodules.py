from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from pytorch_lightning import LightningDataModule

from convexrobust.data.cifar_select import CIFAR10SelectDataModule
from convexrobust.data.mnist_select import MNISTSelectDataModule
from convexrobust.data.fashion_mnist_select import FashionMNISTSelectDataModule
from convexrobust.data.malimg import MalimgDataModule
from convexrobust.data.circles import CirclesDataModule
from convexrobust.utils import torch_utils as TU

import types
from typing import Optional

names = [
    'mnist_38', 'fashion_mnist_shirts', 'malimg', 'cifar10_catsdogs',
    'cifar10_dogscats', 'circles'
]

null_transform_mnist = transforms.Compose([transforms.ToTensor()])
null_transform_fashion_mnist = transforms.Compose([transforms.ToTensor()])
null_transform_malimg = transforms.Compose([
    transforms.CenterCrop(512),
    transforms.ToTensor(),
])
null_transform_cifar10 = transforms.Compose([transforms.ToTensor()])

transforms_mnist = {
    'train_transforms': transforms.Compose([
        transforms.RandomCrop(28, padding=1, padding_mode='edge'),
        transforms.ToTensor(),
    ]),
    'val_transforms': null_transform_mnist,
    'test_transforms': null_transform_mnist
}
transforms_fashion_mnist = {
    'train_transforms': transforms.Compose([
        transforms.RandomCrop(28, padding=1, padding_mode='edge'),
        transforms.ToTensor(),
    ]),
    'val_transforms': null_transform_mnist,
    'test_transforms': null_transform_mnist
}
transforms_malimg = {
    'train_transforms': transforms.Compose([
        transforms.CenterCrop(512),
        transforms.RandomCrop(512, padding=20),
        transforms.ToTensor(),
    ]),
    'val_transforms': null_transform_malimg,
    'test_transforms': null_transform_malimg
}
transforms_cifar10 = {
    'train_transforms': transforms.Compose([
        transforms.RandomCrop(32, padding=3, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val_transforms': null_transform_cifar10,
    'test_transforms': null_transform_cifar10
}

def make_null_transforms(null_transform):
    return {
        'train_transforms': null_transform,
        'val_transforms': null_transform,
        'test_transforms': null_transform
    }

null_transforms_mnist = make_null_transforms(null_transform_mnist)
null_transforms_fashion_mnist = make_null_transforms(null_transform_fashion_mnist)
null_transforms_malimg = make_null_transforms(null_transform_malimg)
null_transforms_cifar10 = make_null_transforms(null_transform_cifar10)


# If change, update in l infty nets dataset.py

_MNIST_38_MEAN = [0.1457]
_MNIST_38_STDDEV = [0.3215]

_FASHION_MNIST_SHIRTS_MEAN = [0.3294]
_FASHION_MNIST_SHIRTS_STDDEV = [0.3452]

_MALIMG_MEAN = [0.1857]
_MALIMG_STDDEV = [0.3029]

_CIFAR10_CATSDOGS_MEAN = [0.4986, 0.4610, 0.4165]
_CIFAR10_CATSDOGS_STDDEV = [0.2542, 0.2482, 0.2534]

all_transforms = {
    'mnist_38': transforms_mnist, 'fashion_mnist_shirts': transforms_fashion_mnist,
    'malimg': transforms_malimg, 'cifar10_catsdogs': transforms_cifar10,
    'cifar10_dogscats': transforms_cifar10, 'circles': None
}
null_transforms = {
    'mnist_38': null_transforms_mnist, 'fashion_mnist_shirts': null_transforms_fashion_mnist,
    'malimg': null_transforms_malimg, 'cifar10_catsdogs': null_transforms_cifar10,
    'cifar10_dogscats': null_transforms_cifar10, 'circles': None
}
data_in_n = {
    'mnist_38': 1 * 28 * 28, 'fashion_mnist_shirts': 1 * 28 * 28,
    'malimg': 1 * 512 * 512, 'cifar10_catsdogs': 3 * 32 * 32,
    'cifar10_dogscats': 3 * 32 * 32
}


def get_datamodule(
        name: str, eval_n: Optional[int]=None, batch_size: Optional[int]=None, no_transforms=False,
        normalize=False, normalize_average_stddev=False,
        labels_override: Optional[list[int]]=None, datamodule_args={}
    ) -> LightningDataModule:

    """Return the datamodule associated to a string name. Datamodules MUST have an in_n attribute
    specifying dimensionality. Furthermore, the certification code expects a batch size of 1 from
    the test dataloader, so this should also be followed.

    Args:
        name (str): The name of the datamodule to fetch.
        eval_n: (int, optional): If specified, adds an eval_iterator method that iterates over
            the first eval_n items in the test dataloader.
        batch_size (int, optional): Option to specify an explicit batch size for training and
            validation. Test batch size should remain 1. Defaults to None.
        no_transforms (bool, optional): Ignore training transforms. Used for convex separability
            test. Defaults to False.
        normalize (bool, optional): Whether to add a normalization layer. Defaults to False.
        normalize_average_stddev (bool, optional): Normalize the standard deviation of all channels
            by a fixed scalar. Used for l infinity distance nets. Defaults to False.
        labels_override (list[int], optional): Overrides the label selection.
        datamodule_args: passed to datamodule

    Raises:
        ValueError: If can't find datamodule for specified name.

    Returns:
        LightningDataModule: The requested datamodule.
    """
    assert not (normalize_average_stddev and not normalize)

    if batch_size is None:
        batch_size = 32 if name=='malimg' else 64
    fixed_params = {
        'batch_size': batch_size, 'num_workers': 0, 'shuffle': True, 'drop_last': True,
        **datamodule_args
    }

    trans = null_transforms[name] if no_transforms else all_transforms[name]
    if normalize:
        for (_, transform) in trans.items():
            transform.transforms.append(_normalize_transform(name, normalize_average_stddev))

    def labels_default(labels):
        return labels_override if labels_override is not None else labels

    if name == 'mnist_38':
        datamodule = MNISTSelectDataModule(labels=labels_default([3, 8]), **fixed_params, **trans)
    elif name == 'fashion_mnist_shirts':
        datamodule = FashionMNISTSelectDataModule(
            labels=labels_default([0, 6]), **fixed_params, **trans
        )
    elif name == 'malimg':
        datamodule = MalimgDataModule(**fixed_params, **trans)
    elif name in ['cifar10_catsdogs', 'cifar10_dogscats']:
        labels = labels_default([3, 5] if name == 'cifar10_catsdogs' else [5, 3])
        datamodule = CIFAR10SelectDataModule(labels=labels, **fixed_params, **trans)
    elif name == 'circles':
        datamodule = CirclesDataModule()
    else:
        raise ValueError()

    datamodule.prepare_data()
    datamodule.setup()

    datamodule.name = name
    datamodule.in_n = data_in_n[name]
    if eval_n is not None:
        def eval_iterator(self, do_tqdm=False):
            return TU.fetch_dataloader(self.test_dataloader(), eval_n, do_tqdm)
        datamodule.eval_iterator = types.MethodType(eval_iterator, datamodule)

    return datamodule


def _normalize_transform(name: str, average_stddev=False) -> transforms.Normalize:
    """A dataloader normalization transform with scalar std. Used for l infinity distance nets."""
    stddev_process = lambda std: [np.mean(std)] * len(std) if average_stddev else lambda std: std

    if name == 'mnist_38':
        return transforms.Normalize(_MNIST_38_MEAN, stddev_process(_MNIST_38_STDDEV))
    elif name == 'fashion_mnist_shirts':
        return transforms.Normalize(
            _FASHION_MNIST_SHIRTS_MEAN, stddev_process(_FASHION_MNIST_SHIRTS_STDDEV)
        )
    elif name == 'malimg':
        return transforms.Normalize(_MALIMG_MEAN, stddev_process(_MALIMG_STDDEV))
    elif name in ['cifar10_catsdogs', 'cifar10_dogscats']:
        return transforms.Normalize(_CIFAR10_CATSDOGS_MEAN, stddev_process(_CIFAR10_CATSDOGS_STDDEV))
    raise ValueError('Invalid dataset selected')


class NormalizeLayer(torch.nn.Module):
    # Original source from https://github.com/locuslab/smoothing

    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be
      the first layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: list[float], sds: list[float], mean_only, average_stddev):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = nn.Parameter(torch.tensor(means), requires_grad=False)
        self.sds = nn.Parameter(torch.tensor(sds), requires_grad=False)
        if average_stddev:
            self.sds.fill_(self.sds.mean())
        if mean_only:
            self.sds.fill_(1.0)

        self.vars = nn.Parameter(self.sds.data.pow(2), requires_grad=False)

        self.weight = nn.Parameter(torch.ones_like(self.means.data), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros_like(self.means.data), requires_grad=False)

    def forward(self, input: torch.Tensor):
        # Had to do some janky adjusting to make work with ab-crown
        new_normalized = F.batch_norm(
            input, self.means.data, self.vars.data, momentum=1.0, eps=0.0,
            weight=self.weight, bias=self.bias
        )

        # (batch_size, num_channels, height, width) = input.shape
        # means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        # sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        # old_normalized = (input - means) / sds
        # assert (old_normalized - new_normalized).abs().max() < 0.0001

        return new_normalized


def get_normalize_layer(name: str, mean_only=False, average_stddev=False) -> NormalizeLayer:
    """Gets a normalization layer for a datamodule that can be used in a model.

    Args:
        name (str): The name of the datamodule to normalize.
        mean_only (bool, optional): Whether to only normalize the mean and not scale the
            standard deviation. Defaults to False.
        average_stddev (bool, optional): If scaling by the standard deviation, average to make
            it a scalar across all channels. Defaults to False.

    Raises:
        ValueError: If can't get normalization layer for specified name

    Returns:
        NormalizeLayer: The normalization layer.
    """
    args = {'mean_only': mean_only, 'average_stddev': average_stddev}
    if name == 'mnist_38':
        return NormalizeLayer(_MNIST_38_MEAN, _MNIST_38_STDDEV, **args)
    elif name == 'fashion_mnist_shirts':
        return NormalizeLayer(_FASHION_MNIST_SHIRTS_MEAN, _FASHION_MNIST_SHIRTS_STDDEV, **args)
    elif name == 'malimg':
        return NormalizeLayer(_MALIMG_MEAN, _MALIMG_STDDEV, **args)
    elif name in ['cifar10_catsdogs', 'cifar10_dogscats']:
        return NormalizeLayer(_CIFAR10_CATSDOGS_MEAN, _CIFAR10_CATSDOGS_STDDEV, **args)
    raise ValueError('Invalid dataset selected')
