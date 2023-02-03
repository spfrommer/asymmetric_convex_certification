from __future__ import annotations

from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import List

from convexrobust.utils import torch_utils


class ConvexModule(nn.Module, ABC):
    """A contract for an input-convex network. After project() is called, the forward method must be
    convex in its input.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """A forwards pass through the network. Should produce a Tensor with one scalar output per
            batched input, each output convex in the input.

        Args:
            x (Tensor): [batch_n x ...]. The input to the network.

        Returns:
            Tensor: [batch_n]. The output, convex in x.
        """
        pass

    @abstractmethod
    def project(self) -> None:
        """Should perform any required weight matrix projections to ensure convexity."""
        pass

    def init_project(self) -> None:
        """An optional overridable special initial projection step. Allows for a custom weight
        initialization."""
        self.project()


def project_weight_positive(weight: Parameter) -> None:
    """Project a weight parameter to be elementwise positive. Used during training."""
    weight.data.clamp_(0.0)


def init_weight_positive(weight: Parameter, linear=False, strategy='uniform') -> None:
    """Initialize a weight parameter to elementwise positive.

    Args:
        weight (Parameter): The parameter to project.
        linear (bool, optional): Whether the weight is a linear layer (instead of convolutional).
            Defaults to False.
        strategy (str, optional): Either 'uniform' or 'simple'. Defaults to 'uniform'.
    """
    if strategy == 'simple':
        weight.data.clamp_(0)
    elif strategy == 'uniform':
        if linear:
            weight.data.uniform_(0.0, 0.003)
        else:
            weight.data.uniform_(0.0, 0.005)


class ConvexMLP(ConvexModule):
    """An input-convex multi layer perceptron, with optional 1d batchnorms and skip connections."""
    def __init__(self, in_n: int, feature_ns: List[int], nonlin=nn.ReLU,
                 skip_connections=True, init_batchnorm=False, batchnorms=True):
        """
        Args:
            in_n (int): The number of inputs to the network for each batch entry.
            feature_ns (List[int]): The list of hidden layer sizes. Output size is always one
                scalar.
            nonlin (type[Module], optional): The nonlinearity module class. Defaults to nn.ReLU.
            skip_connections (bool, optional): Whether to add input skip connections to deeper
                layers. Defaults to True.
            init_batchnorm (bool, optional): Whether to apply an initial batchnorm to the input.
            batchnorms (bool, optional): Whether to apply 1d batchnorms after nonlinearities.
                Defaults to True.
        """
        super().__init__()

        out_n = 1
        feature_ns = [in_n] + feature_ns + [out_n]
        self.layer_n = len(feature_ns) - 1

        self.skip_connections = skip_connections
        self.init_batchnorm = nn.BatchNorm1d(1) if init_batchnorm else None
        self.batchnorms = batchnorms
        self.nonlin = nonlin().to(torch_utils.device())

        W_z, W_x, nonlins, bns = [], [], [], []
        for i, (prev_feature_n, curr_feature_n) in enumerate(zip(feature_ns, feature_ns[1:])):
            W_z.append(nn.Linear(prev_feature_n, curr_feature_n, bias=not skip_connections))
            if self.skip_connections:
                W_x.append(nn.Linear(in_n, curr_feature_n))

            if i < self.layer_n - 1:
                nonlins.append(nonlin().to(torch_utils.device()))
                if batchnorms:
                    bns.append(nn.BatchNorm1d(1))

        self.W_z = nn.ModuleList(W_z).to(torch_utils.device())
        self.W_x = nn.ModuleList(W_x).to(torch_utils.device())
        self.nonlins = nn.ModuleList(nonlins).to(torch_utils.device())
        self.bns = nn.ModuleList(bns).to(torch_utils.device())

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], -1)
        if self.init_batchnorm is not None:
            x = self.init_batchnorm(x.unsqueeze(1)).squeeze(1)

        z = x

        for i in range(self.layer_n):
            if self.skip_connections:
                z = self.W_z[i](z) + self.W_x[i](x)
            else:
                z = self.W_z[i](z)

            if i < self.layer_n - 1:
                z = self.nonlins[i](z)
                if self.batchnorms:
                    z = self.bns[i](z.unsqueeze(1)).squeeze(1)

        return z.squeeze(1)

    def project(self):
        # First layer does not need to be positive as it is an affine transformation of the input.
        for W_z in self.W_z[1:]:
            project_weight_positive(W_z.weight)

        for bn in self.bns:
            bn.weight.data.clamp_(0.001)

    def init_project(self, strategy='uniform'):
        for W_z in self.W_z[1:]:
            init_weight_positive(W_z.weight, linear=True, strategy=strategy)


class ConvexConvNet(ConvexModule):
    """An input-convex convnet, with convolution parameters and optional skip connections."""
    def __init__(self, image_size=224, channel_n=3, feature_n=32, depth=5,
                 conv_1_stride=1, conv_1_kernel_size=15, conv_1_dilation=1,
                 deep_kernel_size=5, pool_size=1, nonlin=nn.ReLU, skip_connections=False):
        """
        Args:
            image_size (int, optional): The width of the square image inputs. Defaults to 224.
            channel_n (int, optional): Number of input channels. If a feature map is being used,
            this should be the number of output channels from the feature map. Defaults to 3.
            feature_n (int, optional): Number of channels for deep network features. Defaults to 32.
            depth (int, optional): After the first convolutional layer, how many additional layers.
                Defaults to 5.
            conv_1_stride (int, optional): Initial convolution stride. Defaults to 1.
            conv_1_kernel_size (int, optional): Initial convolution kernel size. Defaults to 15.
            conv_1_dilation (int, optional): Initial convolution dilation. Defaults to 1.
            deep_kernel_size (int, optional): Kernel size for all layers after the first. Defaults
                to 5.
            pool_size (int, optional): Pooling width for the final pooling operation. Defaults to 1.
            nonlin (type[Module], optional): The nonlinearity module class. Defaults to nn.ReLU.
            skip_connections (bool, optional): Whether to add input skip connections to deeper
                layers. Since the convex blocks have residiual connections, this is not strictly
                necessary as it is for MLPs. Included for completness. Defaults to False.
        """
        super().__init__()

        assert (conv_1_kernel_size % 2) == 1

        conv_1_padding = (conv_1_kernel_size // 2) * conv_1_dilation

        self.bn_1 = nn.BatchNorm2d(channel_n)
        self.conv_1 = nn.Conv2d(
            channel_n, feature_n, kernel_size=conv_1_kernel_size,
            stride=conv_1_stride, dilation=conv_1_dilation, padding=conv_1_padding
        )
        self.nonlin_1 = nonlin()

        self.blocks = nn.ModuleList(
            [ConvexBlock(feature_n, deep_kernel_size, nonlin) for _ in range(depth)]
        )

        self.skip_connections = skip_connections
        if skip_connections:
            self.skips = nn.ModuleList(
                [nn.Conv2d(channel_n, feature_n, kernel_size=deep_kernel_size,
                           padding=deep_kernel_size // 2) for _ in range(depth)]
            )

        self.max_pool = nn.MaxPool2d(pool_size, pool_size)
        self.bn_last = nn.BatchNorm2d(feature_n)

        final_image_size = image_size // (pool_size * conv_1_stride)
        self.readout = nn.Linear(feature_n * (final_image_size ** 2), 1, bias=True)

    def forward(self, x):
        batch_n = x.shape[0]

        x = self.bn_1(x)
        z = self.conv_1(x)
        z = self.nonlin_1(z)

        for i, block in enumerate(self.blocks):
            z = block(z)
            if self.skip_connections:
                z = z + 0.1 * self.skips[i](x)

        z = self.max_pool(z)
        z = self.bn_last(z)
        z = self.readout(z.reshape(batch_n, -1))

        return z.squeeze(1)

    def project(self):
        # conv_1 does not need to be positive as it is an affine transformation of the input.
        for block in self.blocks:
            project_weight_positive(block.conv.weight)
            # Must also make sure batchnorms aren't negating inputs and breaking convexity.
            # Not a practical problem but included to make sure.
            block.bn.weight.data.clamp_(0.001)

        self.bn_last.weight.data.clamp_(0.001)
        project_weight_positive(self.readout.weight)

    def init_project(self, strategy='uniform'):
        for block in self.blocks:
            init_weight_positive(block.conv.weight, strategy=strategy)

        init_weight_positive(self.readout.weight, linear=True, strategy=strategy)


class ConvexBlock(nn.Module):
    """A block in the convex convnet structure. Has a residual connection over the convolution."""
    def __init__(self, channels: int, kernel_size: int, nonlin: type[Module]):
        """
        Args:
            channels (int): Number of input / ouput channels (are the same to allow for residual).
            kernel_size (int): Size of the convolutional kernel.
            nonlin (type[Module]): The nonlinearity class.
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.nonlin = nonlin()

    def forward(self, x):
        z = self.bn(x)
        z = self.conv(z) + z
        z = self.nonlin(z)

        return z


class StandardMLP(nn.Module):
    """A regular multi layer perceptron."""
    def __init__(
            self, in_n: int, out_n: int, feature_ns: List[int], linear=nn.Linear, nonlin=nn.ReLU
        ):

        super().__init__()
        feature_ns = [in_n] + feature_ns
        self.nonlin = nn.ReLU().to(torch_utils.device())

        layers = []
        for (prev_feature_n, curr_feature_n) in zip(feature_ns, feature_ns[1:]):
            layers.append(linear(prev_feature_n, curr_feature_n))
            layers.append(nonlin())
        layers.append(linear(feature_ns[-1], out_n))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)
