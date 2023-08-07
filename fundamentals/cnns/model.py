
import functools
import numpy as np
import torch.nn as nn

from cnns.ops import *


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()

        weight = t.randn((out_features, in_features))
        weight = (2 * weight -1) / np.sqrt(in_features)
        self.weight = nn.Parameter(weight)

        if bias:
            m = t.randn((out_features,))
            m = (2 * m - 1) / np.sqrt(in_features)
            self.bias = nn.Parameter(m)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = einops.einsum(x, self.weight, "... i, j i -> ... j")

        if self.bias is not None:
            x = x + self.bias
        return x

# x = t.randn((3, 4))
# m = Linear(4, 3)
# m(x).shape

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])

# mp = MaxPool2D(kernel_size=3, stride=2, padding=1)

class ReLU(nn.Module): # skip
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.tensor(0.0))

class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        shape = input.shape

        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        shape_left = shape[:start_dim]
        shape_middle = functools.reduce(lambda x, y: x*y, shape[start_dim : end_dim+1])
        shape_right = shape[end_dim+1:]

        new_shape = shape_left + (shape_middle,) + shape_right

        return t.reshape(input, new_shape)

# x = t.randn((2, 3, 4, 5))
# f = Flatten()
# f(x).shape


class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernel_height, kernel_width = force_pair(kernel_size)
        sf = 1 / np.sqrt(in_channels * kernel_width * kernel_height)
        weight = sf * (2 * t.rand(out_channels, in_channels, kernel_height, kernel_width) - 1)

        self.weight = nn.Parameter(weight)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["in_channels", "out_channels", "kernel_size", "stride", "padding"]])

# m = nn.Conv2d(16, 33, (3, 5), )
# x = t.randn(20, 16, 50, 100)
# o = m(x)
# o.shape

# mm = Conv2d(16, 33, (3, 5))
# oo = mm(x)
# oo.shape
# t.testing.assert_close(o, oo)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = ReLU()
        self.flatten = Flatten()
        self.fc = Linear(in_features=14*14*32, out_features=10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# model = SimpleCNN().to(device)