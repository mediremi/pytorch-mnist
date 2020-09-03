import os.path

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
LAYER_SIZE = 64
OUTPUT_SIZE = 10
MODEL_PATH = "model/model.pt"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layers = nn.ModuleList([
            nn.Linear(INPUT_SIZE, LAYER_SIZE),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.Linear(LAYER_SIZE, LAYER_SIZE)
        ])
        self.output_layer = nn.Linear(LAYER_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return F.log_softmax(self.output_layer(x), dim=1)
        return x

def get_dataset(train=False):
    return datasets.MNIST(
        '',
        train=train,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))

def load_data(dataset, shuffle=False, batch_size=10):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle)

def train(model, trainset, epochs=3):
    model.train()
    for epoch in range(epochs):
        for data in trainset:
            X, y = data
            model.zero_grad()
            output = model(X.view(-1, INPUT_SIZE))
            loss = F.nll_loss(output, y)
            loss.backward()
            optimiser.step()
        print(loss)

def evaluate(model, testset):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testset:
            X, y = data
            output = model(X.view(-1, INPUT_SIZE))
            for i, expected in enumerate(output):
                if torch.argmax(expected) == y[i]:
                    correct += 1
                total += 1
    accuracy = correct / total
    return accuracy

def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)

def load_model(model_class):
    model = model_class()
    model.load_state_dict(torch.load(MODEL_PATH))
    return model

training = get_dataset(train=True)
testing = get_dataset()

trainset = load_data(training, shuffle=True)
testset = load_data(testing)

if os.path.isfile(MODEL_PATH):
    net = load_model(Net)
else:
    net = Net()
    optimiser = optim.Adam(net.parameters(), lr=0.001)
    train(net, trainset)
    save_model(net)

accuracy = evaluate(net, testset)
print("Testset accuracy: {}%".format(round(accuracy, 3) * 100))
