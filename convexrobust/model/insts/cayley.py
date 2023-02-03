import torch
import torch.nn as nn

from convexrobust.data import datamodules
from convexrobust.model.modules import StandardMLP
from convexrobust.model.cayley_certifiable import CayleyCertifiable

from lib.orthconv.models import ResNet9
from lib.orthconv.layers import CayleyLinear, GroupSort


class CayleyMnist(CayleyCertifiable):
    def __init__(self, **kwargs):
        super().__init__(
            model=nn.Sequential(
                nn.Flatten(),
                StandardMLP(784, 2, [200, 50], linear=CayleyLinear, nonlin=GroupSort)
            ),
            normalize=datamodules.get_normalize_layer('mnist_38', mean_only=True),
            **kwargs
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return [optimizer]


class CayleyFashionMnist(CayleyCertifiable):
    def __init__(self, **kwargs):
        super().__init__(
            model=ResNet9(in_channels=1, out_n=2, init_pad=True),
            normalize=datamodules.get_normalize_layer('fashion_mnist_shirts', mean_only=True),
            **kwargs
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return [optimizer]


class CayleyCifar(CayleyCertifiable):
    def __init__(self, **kwargs):
        super().__init__(
            model=ResNet9(out_n=2),
            normalize=datamodules.get_normalize_layer('cifar10_catsdogs', mean_only=True),
            **kwargs
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return [optimizer]
