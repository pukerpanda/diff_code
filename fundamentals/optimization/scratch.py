
import torch.nn as nn
import torch as t

from torchvision import models
from torchvision import datasets, transforms, models

from resnet import utils

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

