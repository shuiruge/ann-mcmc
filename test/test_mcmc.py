"""
Description
---------

"""

import sys
sys.path.append('../sample/')
from ann import NeuralNetwork
import numpy as np
from random import gauss, random

import sys
sys.path.append('../../')
import little_mcmc.sample.mcmc as mc
from time import time

import matplotlib.pyplot as plt




# Typing Hint
# ----------------------
from typing import List, Tuple, Mapping

Array = np.array(List[float])
Value = float
# ----------------------


# E.g.
def error_function(outputs: Array, targets: Array) -> float:
    return 0.5 * np.sum((outputs - targets) ** 2)

# Input
x = np.arange(-np.pi, np.pi, 0.1)

# Target
y = np.sin(x)

net_size = [2, 1]
input_size = 1


# Preparing for mcmc
def random_move(net: NeuralNetwork, step_length=0.1) -> NeuralNetwork:
    result_net = net.copy()
    
    for layer in net.layers:
        for perceptron in layer:
            perceptron.weights = (  perceptron.weights
                                  + np.array([ gauss(0, 1) * step_length
                                               for _ in range(len(perceptron.weights))])
                                 )
    
    return net


# Input
x = np.arange(-np.pi, np.pi, 0.3)

# Target
y = np.sin(x)

def f(net: NeuralNetwork, inputs=x, targets=y) -> Value:
        
    outputs = np.array([net.output([__])[0] for __ in inputs])
    
    return -1 * error_function(outputs, targets)


# Do mcmc
chain_num = 100
chain_list = []

t_begin = time()

for step in range(chain_num):
    init_net = NeuralNetwork(net_size, input_size)

    net_chain = mc.single_chain_mcmc(
                     f, random_move, init_net,
                     tolerence=0.01,
                     max_break_counter=30 * np.prod(net_size) * input_size,
                     iterations = 10 ** 10
                     )
    chain_list.append(net_chain)

bc = mc.best_chain(chain_list)
best_net, ef_value = bc[-1][0], bc[-1][1]

t_end = time()
print('time spent: ', t_end - t_begin)


# Show result
print('error_function_value: ', ef_value)

# Plot result
net_y = [best_net.output([__]) for __ in x]

plt.plot(x, y, '-')
plt.plot(x, net_y, 'ro', alpha = 0.3)


# Return
# time spent:  7.975084066390991
# error_function_value:  -3.03309812875

# Not so good, as the plot shows.
