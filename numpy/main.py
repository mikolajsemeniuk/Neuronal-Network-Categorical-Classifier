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
            # single_output: ndarray = single_output.reshape(-1, 1)
            single_output: ndarray = np.expand_dims(single_output, axis = 1)
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

    def __init__(self, learning_rate = 0.1, decay = 0., momentum = 0.):
        self.learning_rate = learning_rate

        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

        self.momentum = momentum

    def update_params(self, layer):

        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                
                # weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
                # print(f'weights: {np.sum(self.momentum * layer.weight_momentums)}')
                # weight_updates = - self.current_learning_rate * layer.dweights
                # layer.weight_momentums = weight_updates
                layer.weights_momentums = -self.current_learning_rate * layer.dweights
                
                # bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
                # print(f'bias: {np.sum(self.momentum * layer.bias_momentums)}')
                # bias_updates = - self.current_learning_rate * layer.dbiases
                # layer.bias_momentums = bias_updates
                layer.bias_momentums = -self.current_learning_rate * layer.dbiases
            else:
                weight_updates = -self.current_learning_rate * layer.dweights
                bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += -self.learning_rate * layer.dweights 
        layer.biases += -self.learning_rate * layer.dbiases

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad:

    def __init__(self, learning_rate=.1, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_cache += layer.dweights**2
            layer.bias_cache += layer.dbiases**2
            layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
            layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
            layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
            # Vanilla SGD parameter update + normalization
            # with square rooted cache
            layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
            layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update_params(self):
        self.iterations += 1

# Adam optimizer
class Optimizer_Adam:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
        beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
            # Update momentum with current gradients
            layer.weight_momentums = self.beta_1 * \
            layer.weight_momentums + \
            (1 - self.beta_1) * layer.dweights
            layer.bias_momentums = self.beta_1 * \
            layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases
            # Get corrected momentum
            # self.iteration is 0 at first pass
            # and we need to start with 1 here
            weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
            bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
            # Update cache with squared current gradients
            layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
            layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
            # Get corrected cache
            weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
            bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
            # Vanilla SGD parameter update + normalization
            # with square rooted cache
            layer.weights += -self.current_learning_rate * \
            weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) +
            self.epsilon)
            layer.biases += -self.current_learning_rate * \
            bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) +
            self.epsilon)
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

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

# optimizer = Optimizer_SGD(learning_rate = .1, decay = 1e3, momentum = 0.5)
optimizer = Optimizer_SGD()
# optimizer = Optimizer_Adagrad(decay=1e-4)
# optimizer = Optimizer_Adam(learning_rate=1, decay=5e-7)

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
      
    optimizer.pre_update_params()

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    optimizer.post_update_params()
# 1
# 0:00:00.172553
# 2
# 0:00:00.063677
print(f'Time taken: {datetime.now() - start}')