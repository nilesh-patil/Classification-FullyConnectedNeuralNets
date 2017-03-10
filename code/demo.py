"""This is a simple example for training a neural network. This example
demonstrates some usage of the modules. You may use this code to test your
implementation.

"""

import mnist_loader
import numpy as np
from graph import Graph
from loss import Euclidean
from network import Network
from optimization import SGD

# Load the MNIST dataset
training_data, validation_data, test_data = load_data_wrapper()

# The network definition of a neural network
graph_config = [
    ("FullyConnected", {"shape": (30, 784)}),
    ("Sigmoid", {}),
    ("FullyConnected", {"shape": (10, 30)}),
    ("Sigmoid", {})
]

learning_rate = 3.0
batch_size = 10
epoch = 25

graph = Graph(graph_config)
loss = Euclidean()
optimizer = SGD(learning_rate=learning_rate,
                batch_size=batch_size)

network = Network(graph)
network.train(training_data = training_data,
              loss = loss,
              optimizer = optimizer,
              test_data = test_data,
              epochs = epoch)

# Test a handwritten digit image
x, y = test_data[0]
y_pred = np.argmax(network.inference(x))
print("The image is {}. Your prediction is {}.".format(y, y_pred))
