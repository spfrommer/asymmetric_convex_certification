from pyexpat import model
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import torchvision

from convexrobust.model.modules import StandardMLP

from convexrobust.data import datamodules
from convexrobust.model.randsmooth_certifiable import RandsmoothCertifiable

from lib.smoothingSplittingNoise.src.lib.wide_resnet import WideResNet


class RandsmoothMnist(RandsmoothCertifiable):
    def __init__(self, **kwargs):
        model = torchvision.models.resnet18(pretrained=False, num_classes=2)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        super().__init__(
            model=model,
            normalize=datamodules.get_normalize_layer('mnist_38'),
            **kwargs
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs_n)
        return [optimizer], [scheduler]


class RandsmoothFashionMnist(RandsmoothCertifiable):
    def __init__(self, **kwargs):
        model = torchvision.models.resnet18(pretrained=False, num_classes=2)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        super().__init__(
            model=model,
            normalize=datamodules.get_normalize_layer('fashion_mnist_shirts'),
            **kwargs
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs_n)
        return [optimizer], [scheduler]


class RandsmoothMalimg(RandsmoothCertifiable):
    def __init__(self, **kwargs):
        model = torchvision.models.resnet18(pretrained=False, num_classes=2)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        super().__init__(
            model=model,
            normalize=datamodules.get_normalize_layer('malimg'),
            **kwargs
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs_n)
        return [optimizer], [scheduler]


class RandsmoothCifar(RandsmoothCertifiable):
    def __init__(self, **kwargs):
        super().__init__(
            model=WideResNet(depth=40, widen_factor=2, num_classes=2),
            normalize=datamodules.get_normalize_layer('cifar10_catsdogs'),
            **kwargs
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs_n)
        return [optimizer], [scheduler]


class RandsmoothSimple(RandsmoothCertifiable):
    def __init__(self, **kwargs):
        super().__init__(
            model=nn.Sequential(nn.Flatten(), StandardMLP(2, 2, [200, 200])),
            normalize=None,
            **kwargs
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]
