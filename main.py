import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets

def get_dataset(train=False):
    return datasets.MNIST(
        '',
        train=train,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))

BATCH_SIZE = 10
def load_data(dataset, shuffle=False):
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

train = get_dataset(train=True)
test = get_dataset()

trainset = load_data(train, shuffle=True)
testset = load_data(test)

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
LAYER_SIZE = 64
OUTPUT_SIZE = 10

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layers = [
            nn.Linear(INPUT_SIZE, LAYER_SIZE),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.Linear(LAYER_SIZE, LAYER_SIZE)
        ]
        self.output_layer = nn.Linear(LAYER_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return F.softmax(self.output_layer(x), dim=1)

net = Net()
optimiser = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, INPUT_SIZE))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimiser.step()
    print(loss)
