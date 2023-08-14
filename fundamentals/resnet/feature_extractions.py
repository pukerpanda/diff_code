


from dataclasses import dataclass
from typing import Type

import torch.nn as nn
import torch as t
from torchvision import models
from resnet import utils
from torchvision import datasets, transforms, models

from resnet.model import ResNet34
from resnet.plotly_utils import plot_train_loss_and_test_accuracy_from_trainer
from resnet.trainer import ConvNetTrainer, ConvNetTrainingArgs


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
	transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

n_classes = 10

resnet = ResNet34()
weights = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

resnet = utils.copy_weights(resnet, weights)
resnet.requires_grad_(False)

resnet.out_layers[-1] = nn.Linear(resnet.out_features_per_group[-1], n_classes)

cifar_trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=IMAGENET_TRANSFORM)
cifar_testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=IMAGENET_TRANSFORM)

@dataclass
class ResNetTrainingArgs():
	batch_size: int = 64
	epochs: int = 3
	optimizer: Type[t.optim.Optimizer] = t.optim.Adam
	learning_rate: float = 1e-3
	n_classes: int = 10
	subset: int = 10

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class ResNetTrainer(ConvNetTrainer):
    def __init__(self, args: ResNetTrainingArgs):
        self.args = args
        self.model = resnet.to(device)
        self.optimizer = args.optimizer(self.model.out_layers[-1].parameters(), lr=args.learning_rate)
        self.trainset, self.testset = cifar_trainset, cifar_testset
        self.logged_variables = {"loss": [], "accuracy": []}

    def train_dataloader(self):
        self.model.train()
        return super().train_dataloader()

    def val_dataloader(self):
        self.model.eval()
        return super().val_dataloader()

args = ResNetTrainingArgs()
trainer = ResNetTrainer(args)
trainer.train()
plot_train_loss_and_test_accuracy_from_trainer(trainer, title="Feature extraction with ResNet34")
