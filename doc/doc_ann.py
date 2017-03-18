
"""
Description
---------
Documentation of `sample/ann.py`.
"""


import sys
sys.path.append('../sample/')
import ann
import numpy as np


# Perceptron
# Initialize a perceptron with:
#     3 inputs
p = ann.Perceptron(size=3)

# A perceptron has three attributes
# 1. the length of inputs: size
print(p.size)

# 2. the transform function: trans_function
print(p.trans_function)

# 3. weights
print(p.weights)

# A perceptron has one method: output
# Inputs of a perceptron is of type of numpy.ndarray.
inputs = np.array([1, 2, 3])
print(p.output(inputs))


# NeuralNetwork
# Initialize a neural network with:
#     1 hidden layer, involving 2 perceptrons;
#     1 output layer, involving 1 perceptron
#     3 inputs
net = ann.NeuralNetwork([2, 1], 3)

# asdf
print(net.input_size)
print(net.size)
print(net.layers)
print(net.output)
