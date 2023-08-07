
import torch as t
from torch import Tensor
from torch import nn
import torchinfo

from cnns.model import *

class ConvNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)

		self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=2, stride=2, padding=0)

		self.flatten = Flatten()
		self.fc1 = Linear(in_features=7*7*64, out_features=128)
		self.fc2 = Linear(in_features=128, out_features=10)
		self.relu3 = ReLU()

	def forward(self, x: t.Tensor) -> t.Tensor:
		x = self.maxpool1(self.relu1(self.conv1(x)))
		x = self.maxpool2(self.relu2(self.conv2(x)))
		x = self.fc2(self.relu3(self.fc1(self.flatten(x))))
		return x

class ConvNet1(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()

        self.conv1 = Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool2 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = Flatten()
        self.linear1 = Linear(64 * 7 * 7, 128)
        self.relu = ReLU()
        self.linear2 = Linear(128, num_classes)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# m = ConvNet()

# torchinfo.summary(m, (1, 1, 28, 28))