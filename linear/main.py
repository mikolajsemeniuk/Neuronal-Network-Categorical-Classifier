# conda activate torch_cuda

import torch

from numpy import ndarray
from sklearn.datasets import load_boston

X: ndarray
y: ndarray
X, y = load_boston(return_X_y = True)


from sklearn.preprocessing import StandardScaler
X: ndarray = \
    StandardScaler().fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=80718)


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
preds: ndarray = \
    regression.predict(X_test)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print(f'mae: {mean_absolute_error(preds, y_test)}')
print(f'mse: {mean_squared_error(preds, y_test, squared=False)}')
print(f'r2: {r2_score(preds, y_test)}')


from torch import Tensor
x_data: Tensor = \
    torch.from_numpy(X_train).float()
y_data: Tensor = \
    torch.from_numpy(y_train).float()


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize: int, outputSize: int):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
        print(type(self.linear))

    def forward(self, x):
        out = self.linear(x)
        return out

 
our_model = linearRegression(13, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)
 
for epoch in range(570):
    
    optimizer.zero_grad()

    pred_y = our_model(x_data)

    loss = criterion(pred_y, y_data)
 
    loss.backward()

    optimizer.step()
    #print('epoch {}, loss {}'.format(epoch, loss.item()))

preds: ndarray = \
    our_model(torch.from_numpy(X_test).float()).detach().numpy()
print(f'mae: {mean_absolute_error(preds, y_test)}')
print(f'mse: {mean_squared_error(preds, y_test, squared=False)}')
print(f'r2: {r2_score(preds, y_test)}')
