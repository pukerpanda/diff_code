
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
import torch as t
from torch.functional import F
from tqdm import tqdm
from resnet.model import ConvNet
from resnet.utils import get_mnist

from resnet.plotly_utils import line, plot_train_loss_and_test_accuracy_from_trainer


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

model = ConvNet().to(device)

batch_size = 64
epochs = 3

mnist_trainset, _ = get_mnist(subset = 10)
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)

optimizer = t.optim.Adam(model.parameters())
loss_list = []

for epoch in tqdm(range(epochs)):
    for imgs, labels in mnist_trainloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())

line(loss_list,
    yaxis_range=[0, max(loss_list) + 0.1],
    labels={"x": "Num batches seen", "y": "Cross entropy loss"},
    title="ConvNet training on MNIST",
    width=700)

