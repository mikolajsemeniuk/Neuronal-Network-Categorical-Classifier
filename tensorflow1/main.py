import numpy as np
from numpy import ndarray
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.preprocessing import LabelBinarizer
import tensorflow.compat.v1 as tf
from tensorflow.python.framework.ops import Tensor

tf.disable_v2_behavior()


iris = load_iris()
inputs: ndarray = iris.data
targets: ndarray = \
    LabelBinarizer().fit_transform(iris.target)

inputs_train: ndarray
inputs_test: ndarray
targets_train: ndarray
targets_test: ndarray

inputs_train, inputs_test, targets_train, targets_test = \
    train_test_split(inputs, targets, test_size = 0.3, random_state = 80718)


X: Tensor = tf.placeholder("float", shape=[None, 4])
y: Tensor = tf.placeholder("float", shape=[None, 3])

weights_1: Tensor = tf.Variable(tf.random_normal([4, 64]))
weights_2: Tensor = tf.Variable(tf.random_normal([64, 3]))
biases_1: Tensor = tf.Variable(tf.random_normal([64])),
biases_2: Tensor = tf.Variable(tf.random_normal([3]))


def forward(x) -> Tensor:
    layer_1: Tensor = tf.nn.relu(tf.add(tf.matmul(x, weights_1), biases_1))
    return tf.add(tf.matmul(layer_1, weights_2), biases_2)


predictions: Tensor = forward(X)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = predictions))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)


init = tf.global_variables_initializer()
start = datetime.now()

with tf.Session() as session:
    session.run(init)
    
    for epoch in range(101):
    
        session.run(train_op, feed_dict={X: inputs_train, y: targets_train})
        
        accuracy = np.mean(np.argmax(targets_train, axis = 1) == \
            session.run(tf.argmax(predictions, axis = 1), feed_dict={ X: inputs_train, y: targets_train }))
        
        print(f"Epoch = {epoch}, accuracy = {accuracy:.4f}")

    session.close()

print(f'Time taken: {datetime.now() - start}')
