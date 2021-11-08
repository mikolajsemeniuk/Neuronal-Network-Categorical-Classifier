import numpy as np
from numpy import ndarray

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
inputs: ndarray = iris.data
targets: ndarray = iris.target

from sklearn.preprocessing import LabelBinarizer
targets: ndarray = \
    LabelBinarizer().fit_transform(targets)

inputs_train: ndarray
inputs_test: ndarray
targets_train: ndarray
targets_test: ndarray

inputs_train, inputs_test, targets_train, targets_test = \
    train_test_split(inputs, targets, test_size = 0.3, random_state = 80718)

import tensorflow.compat.v1 as tf
from tensorflow.python.framework.ops import Tensor
tf.disable_v2_behavior()


X: Tensor = tf.placeholder("float", shape=[None, inputs_train.shape[1]])
y: Tensor = tf.placeholder("float", shape=[None, targets_train.shape[1]])

weights = {
  'h1': tf.Variable(tf.random_normal([inputs_train.shape[1], 64])),
  'h2': tf.Variable(tf.random_normal([64, 3]))
}
biases = {
  'b1': tf.Variable(tf.random_normal([64])),
  'b2': tf.Variable(tf.random_normal([3]))
}


def forward(x) -> Tensor:

    layer_1: Tensor = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1: Tensor = tf.nn.relu(layer_1)
    
    layer_2: Tensor = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2: Tensor = tf.nn.softmax(layer_2)
    return layer_2


predictions: Tensor = forward(X)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = predictions))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)


init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    
    for epoch in range(101):
    
        session.run(train_op, feed_dict={X: inputs_train, y: targets_train})
        
        accuracy = np.mean(np.argmax(targets_train, axis = 1) == \
            session.run(tf.argmax(predictions, axis = 1), feed_dict={ X: inputs_train, y: targets_train }))
        
        print(f"Epoch = {epoch}, accuracy = {format(accuracy, '.4f')}")

    session.close()
