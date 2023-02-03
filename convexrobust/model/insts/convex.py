import torch
from torch.optim.lr_scheduler import ExponentialLR

import math

from convexrobust.data import datamodules
from convexrobust.model.modules import ConvexConvNet, ConvexMLP
from convexrobust.model.convex_certifiable import ConvexCertifiable
from convexrobust.model.certificate import Norm


class ConvexMnist(ConvexCertifiable):
    def __init__(self, mlp_params={}, generic_normalize=False, **kwargs):
        default_params = {
            'feature_ns': [200, 50], 'skip_connections': True, 'batchnorms': True
        }
        combined_params = {**default_params, **mlp_params}  # potentially overwrite defaults

        super().__init__(
            convex_model=ConvexMLP(in_n=784, **combined_params),
            **kwargs
        )

        if generic_normalize:
            # Normalize by the average over all MNIST images, not just 3-8
            self.normalize = datamodules.NormalizeLayer(
                means=[0.1307], sds=[0.3081], mean_only=True, average_stddev=False
            )
        else:
            self.normalize = datamodules.get_normalize_layer('mnist_38', mean_only=True)

        self.save_hyperparameters()

    def lipschitz_forward(self, x):
        x = self.normalize(x)
        return x, {Norm.L1: 1, Norm.L2: 1, Norm.LInf: 1}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]


class ConvexFashionMnist(ConvexCertifiable):
    def __init__(self, convnet_params={}, **kwargs):
        default_params = {
            'feature_n': 4, 'depth': 2,
            'conv_1_stride': 1, 'conv_1_kernel_size': 5, 'conv_1_dilation': 1,
            'deep_kernel_size': 3, 'pool_size': 1
        }
        combined_params = {**default_params, **convnet_params}  # potentially overwrite defaults

        super().__init__(
            convex_model=ConvexConvNet(image_size=28, channel_n=1, **combined_params),
            **kwargs
        )

        self.normalize = datamodules.get_normalize_layer('fashion_mnist_shirts', mean_only=True)
        self.save_hyperparameters()

    def lipschitz_forward(self, x):
        x = self.normalize(x)
        return x, {Norm.L1: 1, Norm.L2: 1, Norm.LInf: 1}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]


class ConvexMalimg(ConvexCertifiable):
    def __init__(self, mlp_params={}, **kwargs):
        default_params = {
            'feature_ns': [200, 50], 'skip_connections': True, 'batchnorms': True
        }
        combined_params = {**default_params, **mlp_params}  # potentially overwrite defaults

        super().__init__(
            convex_model=ConvexMLP(in_n=512 * 512, **combined_params),
            **kwargs
        )

        self.normalize = datamodules.get_normalize_layer('malimg', mean_only=True)
        self.save_hyperparameters()

    def lipschitz_forward(self, x):
        x = self.normalize(x)
        return x, {Norm.L1: 1, Norm.L2: 1, Norm.LInf: 1}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]


class ConvexCifar(ConvexCertifiable):
    def __init__(self, convnet_params={}, apply_feature_map=True, **kwargs):
        default_params = {
            'feature_n': 16, 'depth': 4,
            'conv_1_stride': 1, 'conv_1_kernel_size': 11, 'conv_1_dilation': 1,
            'deep_kernel_size': 3, 'pool_size': 1
        }
        combined_params = {**default_params, **convnet_params}  # potentially overwrite defaults
        self.apply_feature_map = apply_feature_map
        channel_n = 6 if apply_feature_map else 3

        super().__init__(
            convex_model=ConvexConvNet(image_size=32, channel_n=channel_n, **combined_params),
            **kwargs
        )

        self.normalize = datamodules.get_normalize_layer('cifar10_catsdogs', mean_only=True)
        self.save_hyperparameters()

    def lipschitz_forward(self, x):
        x = self.normalize(x)
        if self.apply_feature_map:
            return torch.cat([x, x.abs()], dim=1), {Norm.L1: 2, Norm.L2: math.sqrt(2), Norm.LInf: 1}
        return x, {Norm.L1: 1, Norm.L2: 1, Norm.LInf: 1}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.001, momentum=0.9
        )
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]


class ConvexSimple(ConvexCertifiable):
    def __init__(self, **kwargs):
        super().__init__(
            convex_model=ConvexMLP(
                in_n=4, features_n=[50], batchnorms=True, skip_connections=True,
            ),
            **kwargs
        )

        self.save_hyperparameters()

    def lipschitz_forward(self, x):
        x = torch.cat([x, x.abs()], dim=1)
        return x, {Norm.L1: 2, Norm.L2: math.sqrt(2), Norm.LInf: 1}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]
