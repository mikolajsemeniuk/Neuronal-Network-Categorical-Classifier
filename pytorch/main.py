import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from datetime import datetime

from numpy import ndarray
from pandas import DataFrame
from typing import List, Dict, Tuple, Callable, Any

iris = load_iris()
inputs: ndarray = iris.data
targets: ndarray = iris.target

inputs_train: ndarray
inputs_test: ndarray
targets_train: ndarray
targets_test: ndarray

inputs_train, inputs_test, targets_train, targets_test = \
    train_test_split(inputs, targets, test_size = 0.3, random_state = 80718)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(4, 64)
        self.linear_2 = nn.Linear(64, 3)

    def forward(self, inputs: Tensor):
        inputs = self.linear_1(inputs)
        inputs = F.relu(inputs)
        inputs = self.linear_2(inputs)
        return nn.Softmax(dim=1)(inputs)


model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

X_train: Tensor = torch.FloatTensor(inputs_train)
X_test: Tensor = torch.FloatTensor(inputs_test)
y_train: Tensor = torch.LongTensor(targets_train)
y_test: Tensor = torch.LongTensor(targets_test)

start = datetime.now()

for epoch in range(101):
    optimizer.zero_grad()

    predictions: Tensor = \
        model.forward(X_train)

    loss: Tensor = \
        criterion(predictions, y_train)

    accuracy: Tensor = \
        (torch.argmax(predictions, dim = -1) == y_train).sum() / X_train.shape[0]

    loss.backward()
    optimizer.step()

    print(f'epoch {epoch}, loss {format(loss.item(), ".4f")}, accuracy: {format(accuracy, ".4f")}')

print(f'Time taken: {datetime.now() - start}')