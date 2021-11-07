import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from sklearn.datasets import load_iris
from numpy import ndarray
from sklearn.model_selection import train_test_split


iris = load_iris()
inputs: ndarray = iris.data
targets: ndarray = iris.target

inputs_train: ndarray
inputs_test: ndarray
targets_train: ndarray
targets_test: ndarray

inputs_train, inputs_test, targets_train, targets_test = \
    train_test_split(inputs, targets, test_size = 0.3, random_state = 80718)


class NeuronalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(4, 128)
        self.linear_2 = nn.Linear(128, 3)

    def forward(self, inputs: Tensor):
        inputs = self.linear_1(inputs)
        inputs = F.relu(inputs)
        inputs = self.linear_2(inputs)
        return nn.Softmax(dim=1)(inputs)


network = NeuronalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr = 0.01)

X_train: Tensor = torch.FloatTensor(inputs_train)
X_test: Tensor = torch.FloatTensor(inputs_test)
y_train: Tensor = torch.LongTensor(targets_train)
y_test: Tensor = torch.LongTensor(targets_test)


for epoch in range(101):
    optimizer.zero_grad()

    predictions: Tensor = \
        network.forward(X_train)

    loss: Tensor = \
        criterion(predictions, y_train)

    accuracy: Tensor = \
        (torch.argmax(predictions, dim = -1) == y_train).sum() / X_train.shape[0]

    loss.backward()
    optimizer.step()

    print(f'epoch {epoch}, loss {loss.item()}, accuracy: {accuracy}')
