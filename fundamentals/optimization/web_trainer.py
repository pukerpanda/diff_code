from dataclasses import dataclass
from typing import Optional, Tuple, Type
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
from optimization.trainer import ResNetTrainingArgs, get_cifar, test_resnet_on_random_input
from resnet.model import ResNet34

from resnet.plotly_utils import bar, imshow, plot_train_loss_and_test_accuracy_from_trainer

from resnet import utils

import wandb

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

@dataclass
class ResNetTrainingArgsWandb(ResNetTrainingArgs):
    wandb_project: Optional[str] = 'day4-resnet'
    wandb_name: Optional[str] = None


class ResNetTrainerWandb:
    def __init__(self, args: ResNetTrainingArgsWandb):
        self.args = args
        self.model = get_resnet_for_feature_extraction(args.n_classes).to(device)
        self.optimizer = args.optimizer(self.model.out_layers[-1].parameters(), lr=args.learning_rate)
        self.trainset, self.testset = get_cifar(subset=args.subset)
        self.step = 0
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
        wandb.watch(self.model.out_layers[-1], log="all", log_freq=20)

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
        self.step += 1
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
                wandb.log({"loss": loss.item()}, step=self.step)
                desc = f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}, Accuracy = {accuracy:.2f}"
                progress_bar.set_description(desc)
                progress_bar.update()

            # Compute accuracy by summing n_correct over all batches, and dividing by number of items
            accuracy = sum(self.validation_step(imgs, labels) for imgs, labels in self.val_dataloader()) / len(self.testset)
            wandb.log({"accuracy": accuracy.item()}, step=self.step)

        wandb.finish()

# args = ResNetTrainingArgsWandb()
# trainer = ResNetTrainerWandb(args)
# trainer.train()

#test_resnet_on_random_input(trainer.model)
# test_resnet_on_random_input(trainer.model)
#7e870267baa24eb21e3982e34c55d6fcaf1e647f

sweep_config = dict()
# FLAT SOLUTION
# YOUR CODE HERE - fill `sweep_config`
sweep_config = dict(
    method = 'random',
    metric = dict(name = 'accuracy', goal = 'maximize'),
    parameters = dict(
        batch_size = dict(values = [32, 64, 128, 256]),
        epochs = dict(min = 1, max = 4),
        learning_rate = dict(max = 0.1, min = 0.0001, distribution = 'log_uniform_values'),
    )
)

class ResNetTrainerWandbSweeps(ResNetTrainerWandb):
    '''
    New training class made specifically for hyperparameter sweeps, which overrides the values in `args` with
    those in `wandb.config` before defining model/optimizer/datasets.
    '''
    def __init__(self, args: ResNetTrainingArgsWandb):
        wandb.init(project=args.wandb_project, name=args.wandb_name)
        args.batch_size = wandb.config["batch_size"]
        args.epochs = wandb.config["epochs"]
        args.learning_rate = wandb.config["learning_rate"]
        self.args = args
        self.model = get_resnet_for_feature_extraction(args.n_classes).to(device)
        self.optimizer = args.optimizer(self.model.out_layers[-1].parameters(), lr=args.learning_rate)
        self.trainset, self.testset = get_cifar(subset=args.subset)
        self.step = 0
        wandb.watch(self.model.out_layers[-1], log="all", log_freq=20)

def train():
    args = ResNetTrainingArgsWandb()
    trainer = ResNetTrainerWandbSweeps(args)
    trainer.train()

sweep_id = wandb.sweep(sweep=sweep_config, project='part4-optimization-resnet-sweep')
wandb.agent(sweep_id=sweep_id, function=train, count=3)
wandb.finish()
