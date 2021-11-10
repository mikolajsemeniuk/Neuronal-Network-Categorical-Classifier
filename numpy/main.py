import numpy as np
from numpy import ndarray
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from datetime import datetime
from numpy import ndarray

class Layer_Dense:
      
    def __init__(self, n_inputs, n_neurons) -> None:
        # np.random.rand is for Uniform distribution (in the half-open interval [0.0, 1.0))
        # np.random.randn is for Standard Normal (aka. Gaussian) distribution (mean 0 and variance 1)
        
        # lack of initialization
        # self.weights: ndarray = np.random.randn(n_inputs, n_neurons)
        
        # small weights
        # self.weights: ndarray = np.random.randn(n_inputs, n_neurons)

        # he/kaiming initialization
        self.weights: ndarray = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)

        # LeCun/Glorot/Xavier
        self.biases: ndarray = np.zeros((1, n_neurons))
      
    def forward(self, inputs: ndarray) -> None:
        self.inputs: ndarray = inputs
        self.output: ndarray = inputs @ self.weights + self.biases
      
    def backward(self, dvalues: ndarray) -> None:
        self.dweights: ndarray = self.inputs.T @ dvalues
        self.dbiases: ndarray = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs: ndarray = dvalues @ self.weights.T


class Activation_ReLU:

    def forward(self, inputs: ndarray) -> None:
        self.inputs: ndarray = inputs
        self.output: ndarray = np.maximum(0, inputs)
    
    def backward(self, dvalues) -> None:
        self.dinputs: ndarray = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    
    def forward(self, inputs: ndarray) -> None:
        self.inputs: ndarray = inputs
        exp_values: ndarray = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
        self.output: ndarray = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues: ndarray) -> None:
        self.dinputs: ndarray = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output: ndarray = single_output.reshape(-1, 1)
            jacobian_matrix: ndarray = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss_CategoricalCrossentropy:

    def calculate(self, output: ndarray, y: ndarray) -> ndarray:
        sample_losses: ndarray = self.forward(output, y)
        return np.mean(sample_losses)

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    def __init__(self):
        self.activation = Activation_Softmax() 
        self.loss = Loss_CategoricalCrossentropy()
    
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:

    def __init__(self, learning_rate = 0.1):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights 
        layer.biases += -self.learning_rate * layer.dbiases


iris = load_iris()
inputs: ndarray = iris.data
targets: ndarray = iris.target

inputs_train: ndarray
inputs_test: ndarray
targets_train: ndarray
targets_test: ndarray

inputs_train, inputs_test, targets_train, targets_test = \
    train_test_split(inputs, targets, test_size = 0.3, random_state = 80718)


dense1 = Layer_Dense(4, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)

activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_SGD(learning_rate = 0.1)

start = datetime.now()

for epoch in range(101):
    
    dense1.forward(inputs_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    
    # 1
    # activation2.forward(dense2.output)
    # loss = loss_function.calculate(activation2.output, targets_train)
    # 2
    loss = loss_activation.forward(dense2.output, targets_train)

    # 1
    # predictions = np.argmax(activation2.output, axis=1)
    # 2
    predictions = np.argmax(loss_activation.output, axis=1)
    
    if len(targets_train.shape) == 2:
        targets_train = np.argmax(targets_train, axis=1)
    accuracy = np.mean(predictions == targets_train)
    
    print(f'epoch: {epoch}, acc: {accuracy:.4f}, loss: {loss:.4f}')
    
    # 1
    # loss_function.backward(activation2.output, targets_train)
    # activation2.backward(loss_function.dinputs)
    # dense2.backward(activation2.dinputs)

    # 2
    loss_activation.backward(loss_activation.output, targets_train)
    dense2.backward(loss_activation.dinputs)

    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
      
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
# 1
# 0:00:00.172553
# 2
# 0:00:00.063677
print(f'Time taken: {datetime.now() - start}')