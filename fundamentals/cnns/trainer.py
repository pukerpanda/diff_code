

import matplotlib.pyplot as plt
import torch as t

import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cnns.model import SimpleCNN

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_mnist(subset: int = 1):
    mnist_train = datasets.MNIST(root="data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_test = datasets.MNIST(root="data", train=False, download=True, transform=MNIST_TRANSFORM)

    return mnist_train, mnist_test

mnist_train, mnist_test = get_mnist()
mnist_trainloader = DataLoader(mnist_train, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_test, batch_size=64, shuffle=True)

img, label = mnist_train[0]
# imshow(img.squeeze(), cmap="gray")
# plt.show()

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)

img_input = img.unsqueeze(0).to(device)
prob = model(img_input).squeeze().softmax(dim=-1).detach().cpu().numpy()

plt.bar(t.arange(10), prob, )
plt.show()

batch_size = 64
epochs = 3

optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
loss_list = []

import torch.nn.functional as F

for epoch in tqdm(range(epochs)):
    for imgs, labels in mnist_trainloader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())

plt.plot(loss_list)
plt.show()

prob = model(img_input).squeeze().softmax(dim=-1).detach().cpu().numpy()

plt.bar(t.arange(10), prob, )
plt.show()
