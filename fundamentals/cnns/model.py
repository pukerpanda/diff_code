
import functools
import numpy as np
import torch.nn as nn

from cnns.ops import *


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        sf = 1 / np.sqrt(in_features)

        weight = sf * (2 * t.rand(out_features, in_features) - 1)
        self.weight = nn.Parameter(weight)

        if bias:
            bias = sf * (2 * t.rand(out_features,) - 1)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        x = einops.einsum(x, self.weight, "... in_feats, out_feats in_feats -> ... out_feats")
        if self.bias is not None:
            x += self.bias
        return x

    def extra_repr(self) -> str:
        # note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or None, not bool
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

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



# class MaxPool2d(nn.Module):
# 	def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
# 		super().__init__()
# 		self.kernel_size = kernel_size
# 		self.stride = stride
# 		self.padding = padding

# 	def forward(self, x: t.Tensor) -> t.Tensor:
# 		'''Call the functional version of maxpool2d.'''
# 		return maxpool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

# 	def extra_repr(self) -> str:
# 		'''Add additional information to the string representation of this class.'''
# 		return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])



# class ReLU(nn.Module):
# 	def forward(self, x: t.Tensor) -> t.Tensor:
# 		return t.maximum(x, t.tensor(0.0))

# class Flatten(nn.Module):
# 	def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
# 		super().__init__()
# 		self.start_dim = start_dim
# 		self.end_dim = end_dim

# 	def forward(self, input: t.Tensor) -> t.Tensor:
# 		'''
# 		Flatten out dimensions from start_dim to end_dim, inclusive of both.
# 		'''

# 		shape = input.shape

# 		start_dim = self.start_dim
# 		end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

# 		shape_left = shape[:start_dim]
# 		# shape_middle = t.prod(t.tensor(shape[start_dim : end_dim+1])).item()
# 		shape_middle = functools.reduce(lambda x, y: x*y, shape[start_dim : end_dim+1])
# 		shape_right = shape[end_dim+1:]

# 		new_shape = shape_left + (shape_middle,) + shape_right

# 		return t.reshape(input, new_shape)

# 	def extra_repr(self) -> str:
# 		return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])


# class Linear(nn.Module):
# 	def __init__(self, in_features: int, out_features: int, bias=True):
# 		'''
# 		A simple linear (technically, affine) transformation.

# 		The fields should be named `weight` and `bias` for compatibility with PyTorch.
# 		If `bias` is False, set `self.bias` to None.
# 		'''
# 		super().__init__()
# 		self.in_features = in_features
# 		self.out_features = out_features
# 		self.bias = bias

# 		sf = 1 / np.sqrt(in_features)

# 		weight = sf * (2 * t.rand(out_features, in_features) - 1)
# 		self.weight = nn.Parameter(weight)

# 		if bias:
# 			bias = sf * (2 * t.rand(out_features,) - 1)
# 			self.bias = nn.Parameter(bias)
# 		else:
# 			self.bias = None

# 	def forward(self, x: t.Tensor) -> t.Tensor:
# 		'''
# 		x: shape (*, in_features)
# 		Return: shape (*, out_features)
# 		'''
# 		x = einops.einsum(x, self.weight, "... in_feats, out_feats in_feats -> ... out_feats")
# 		if self.bias is not None:
# 			x += self.bias
# 		return x

# 	def extra_repr(self) -> str:
# 		# note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or None, not bool
# 		return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# class Conv2d(nn.Module):
# 	def __init__(
# 		self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
# 	):
# 		'''
# 		Same as torch.nn.Conv2d with bias=False.

# 		Name your weight field `self.weight` for compatibility with the PyTorch version.
# 		'''
# 		super().__init__()
# 		self.in_channels = in_channels
# 		self.out_channels = out_channels
# 		self.kernel_size = kernel_size
# 		self.stride = stride
# 		self.padding = padding

# 		kernel_height, kernel_width = force_pair(kernel_size)
# 		sf = 1 / np.sqrt(in_channels * kernel_width * kernel_height)
# 		weight = sf * (2 * t.rand(out_channels, in_channels, kernel_height, kernel_width) - 1)
# 		self.weight = nn.Parameter(weight)

# 	def forward(self, x: t.Tensor) -> t.Tensor:
# 		'''Apply the functional conv2d you wrote earlier.'''
# 		return conv2d(x, self.weight, self.stride, self.padding)

# 	def extra_repr(self) -> str:
# 		keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
# 		return ", ".join([f"{key}={getattr(self, key)}" for key in keys])

# class SimpleCNN(nn.Module):
# 	def __init__(self):
# 		super().__init__()
# 		self.conv = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
# 		self.maxpool = MaxPool2d(kernel_size=2, stride=2, padding=0)
# 		self.relu = ReLU()
# 		self.flatten = Flatten()
# 		self.fc = Linear(in_features=32*14*14, out_features=10)

# 	def forward(self, x: t.Tensor) -> t.Tensor:
# 		return self.fc(self.flatten(self.relu(self.maxpool(self.conv(x)))))


