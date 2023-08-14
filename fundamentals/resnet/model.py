
import os
from typing import Dict
from jaxtyping import Float, Int
import torch as t
from torch import Tensor
from torch import nn
import torchinfo
from torchvision import datasets, transforms, models

from cnns.model import *
from resnet.utils import copy_weights

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


# m = ConvNet()

# torchinfo.summary(m, (1, 1, 28, 28))

class Sequential(nn.Module):
    _modules: Dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules) # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules) # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for mod in self._modules.values():
            x = mod(x)
        return x

#s = Sequential(Linear(1, 2), Sequential(Linear(2, 3), Linear(3, 4)))

class BatchNorm2d(nn.Module):
	running_mean: Float[Tensor, "num_features"]
	running_var: Float[Tensor, "num_features"]
	num_batches_tracked: Int[Tensor, ""] # This is how we denote a scalar tensor

	def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
		super().__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum

		self.weight = nn.Parameter(t.ones(num_features))
		self.bias = nn.Parameter(t.zeros(num_features))

		self.register_buffer("running_mean", t.zeros(num_features))
		self.register_buffer("running_var", t.ones(num_features))
		self.register_buffer("num_batches_tracked", t.tensor(0))

	def forward(self, x: t.Tensor) -> t.Tensor:
		if self.training:
			mean = t.mean(x, dim=(0, 2, 3), keepdim=True)
			var = t.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)

			self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
			self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
			self.num_batches_tracked += 1
		else:
			mean = einops.rearrange(self.running_mean, "channels -> 1 channels 1 1")
			var = einops.rearrange(self.running_var, "channels -> 1 channels 1 1")

		weight = einops.rearrange(self.weight, "channels -> 1 channels 1 1")
		bias = einops.rearrange(self.bias, "channels -> 1 channels 1 1")

		return ((x - mean) / t.sqrt(var + self.eps)) * weight + bias

	def extra_repr(self) -> str:
		return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])

class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return t.mean(x, dim=(2, 3))

class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        super().__init__()

        self.left = Sequential(
            Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_feats)
        )

        if first_stride > 1:
            self.right = Sequential(
                Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride),
                BatchNorm2d(out_feats)
            )
        else:
            self.right = nn.Identity()

        self.relu = ReLU()

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_left = self.left(x)
        x_right = self.right(x)
        return self.relu(x_left + x_right)

class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        super().__init__()

        blocks = [ResidualBlock(in_feats, out_feats, first_stride)] + [
            ResidualBlock(out_feats, out_feats) for n in range(n_blocks - 1)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        return self.blocks(x)

class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        in_feats0 = 64
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        self.in_layers = Sequential(
            Conv2d(3, in_feats0, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(in_feats0),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        all_in_feats = [in_feats0] + out_features_per_group[:-1]
        self.residual_layers = Sequential(
            *(
                BlockGroup(*args)
                for args in zip(
                    n_blocks_per_group,
                    all_in_feats,
                    out_features_per_group,
                    first_strides_per_group,
                )
            )
        )

        self.out_layers = Sequential(
            AveragePool(),
            Flatten(),
            Linear(out_features_per_group[-1], n_classes),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        x = self.in_layers(x)
        x = self.residual_layers(x)
        x = self.out_layers(x)
        return x


if __name__ == '__main__':
    from PIL import Image
    import os
    import json
    r = ResNet34()
    pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    my_resnet = copy_weights(r, pretrained_resnet)

    IMAGE_FILENAMES = [
        "chimpanzee.jpg",
        "golden_retriever.jpg",
        "platypus.jpg",
        "frogs.jpg",
        "fireworks.jpg",
        "astronaut.jpg",
        "iguana.jpg",
        "volcano.jpg",
        "goofy.jpg",
        "dragonfly.jpg",
    ]

    IMAGE_FOLDER = "resnet_inputs"

    images = [Image.open(os.path.join(IMAGE_FOLDER, filename)) for filename in IMAGE_FILENAMES]

    IMAGE_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    IMAGENET_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    x = t.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0)

    def predict(model, images):
        logits: t.Tensor = model(images)
        return logits.argmax(dim=1)

    with open("imagenet_labels.json") as f:
        imagenet_labels = list(json.load(f).values())

    my_predictions = predict(my_resnet, x)
    pretrained_predictions = predict(pretrained_resnet, x)

    print("\n".join(np.asarray(imagenet_labels)[my_predictions]))

    from resnet.utils import print_param_count

    df = print_param_count(my_resnet, display_df=False)
