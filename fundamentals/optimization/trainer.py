from dataclasses import dataclass
from typing import Tuple, Type
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from torchvision import models
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
from optimization.model import get_resnet_for_feature_extraction

from torch import Tensor, optim
from resnet.model import ResNet34

from resnet.plotly_utils import bar, imshow, plot_train_loss_and_test_accuracy_from_trainer

from resnet import utils

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
	transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def get_cifar(subset: int = 1):
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=IMAGENET_TRANSFORM)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=IMAGENET_TRANSFORM)
    if subset > 1:
        cifar_trainset = Subset(cifar_trainset, indices=range(0, len(cifar_trainset), subset))
        cifar_testset = Subset(cifar_testset, indices=range(0, len(cifar_testset), subset))
    return cifar_trainset, cifar_testset

cifar_trainset, cifar_testset = get_cifar()

# imshow(
#     cifar_trainset.data[:15],
#     facet_col=0,
#     facet_col_wrap=5,
#     facet_labels=[cifar_trainset.classes[i] for i in cifar_trainset.targets[:15]],
#     title="CIFAR-10 images",
#     height=600
# )

@dataclass
class ResNetTrainingArgs():
	batch_size: int = 64
	epochs: int = 3
	optimizer: Type[t.optim.Optimizer] = t.optim.Adam
	learning_rate: float = 1e-3
	n_classes: int = 10
	subset: int = 10


device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class ResNetTrainer:
	def __init__(self, args: ResNetTrainingArgs):
		self.args = args
		self.model = get_resnet_for_feature_extraction(args.n_classes).to(device)
		self.optimizer = args.optimizer(self.model.out_layers[-1].parameters(), lr=args.learning_rate)
		self.trainset, self.testset = get_cifar(subset=args.subset)
		self.logged_variables = {"loss": [], "accuracy": []}

	def _shared_train_val_step(self, imgs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
		imgs = imgs.to(device)
		labels = labels.to(device)
		logits = self.model(imgs)
		return logits, labels

	def training_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
		logits, labels = self._shared_train_val_step(imgs, labels)
		loss = F.cross_entropy(logits, labels)
		loss.backward()
		self.optimizer.step()
		self.optimizer.zero_grad()
		return loss

	@t.inference_mode()
	def validation_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
		logits, labels = self._shared_train_val_step(imgs, labels)
		classifications = logits.argmax(dim=1)
		n_correct = t.sum(classifications == labels)
		return n_correct

	def train_dataloader(self):
		self.model.train()
		return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)

	def val_dataloader(self):
		self.model.eval()
		return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)

	def train(self):
		progress_bar = tqdm(total=self.args.epochs * len(self.trainset) // self.args.batch_size)
		accuracy = t.nan

		for epoch in range(self.args.epochs):

			# Training loop (includes updating progress bar)
			for imgs, labels in self.train_dataloader():
				loss = self.training_step(imgs, labels)
				self.logged_variables["loss"].append(loss.item())
				desc = f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}, Accuracy = {accuracy:.2f}"
				progress_bar.set_description(desc)
				progress_bar.update()

			# Compute accuracy by summing n_correct over all batches, and dividing by number of items
			accuracy = sum(self.validation_step(imgs, labels) for imgs, labels in self.val_dataloader()) / len(self.testset)

			self.logged_variables["accuracy"].append(accuracy.item())

def test_resnet_on_random_input(model: ResNet34, n_inputs: int = 3):
    indices = np.random.choice(len(cifar_trainset), n_inputs).tolist()
    classes = [cifar_trainset.classes[cifar_trainset.targets[i]] for i in indices]
    imgs = cifar_trainset.data[indices]
    device = next(model.parameters()).device
    with t.inference_mode():
        x = t.stack(list(map(IMAGENET_TRANSFORM, imgs)))
        logits: t.Tensor = model(x.to(device))
    probs = logits.softmax(-1)
    if probs.ndim == 1: probs = probs.unsqueeze(0)
    for img, label, prob in zip(imgs, classes, probs):
        imshow(
            img,
            width=200, height=200, margin=0,
            xaxis_visible=False, yaxis_visible=False
        )
        bar(
            prob,
            x=cifar_trainset.classes,
            template="ggplot2", width=600, height=400,
            labels={"x": "Classification", "y": "Probability"},
            text_auto='.2f', showlegend=False,
        )

# args = ResNetTrainingArgs()
# trainer = ResNetTrainer(args)
# trainer.train()
# plot_train_loss_and_test_accuracy_from_trainer(trainer, title="Training ResNet on MNIST data")

# test_resnet_on_random_input(trainer.model)
