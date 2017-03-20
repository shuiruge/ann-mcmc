
"""
Description
---------
Module of construction of ANN. It's written by numpy.
"""

import numpy as np
from random import gauss, uniform
from math import exp
from copy import deepcopy


# Typing Hint
# ----------------------
from typing import List, Callable, Tuple, Mapping, Any

Array = np.array(List[float])
# ----------------------


def sigmoid(x: float) -> float:
    
    return 1 / (1 + exp(-x))


class Perceptron(object):
    """
    Parameters
    ---------
    size:
        The number of in-coming nets to the perceptron, excluding the one for
        threshold.
    """
    
    def __init__(self, size: int) -> None:
        
        self.size = size
        
        # Recall that weights contains threshold, thus len(weights) == size + 1.
        weights = np.array([uniform(-0.5, 0.5) for i in range(size + 1)])
        self.weights = weights
        
        self.trans_function = sigmoid
        
    
    def output(self, inputs: Array) -> float:
        
        assert len(inputs) == self.size, "len(inputs) of a perceptron shall equal to its size."
        
        inputs0 = [__ for __ in inputs] + [1]
        inputs0 = np.array(inputs0)
        
        net = np.dot(self.weights, inputs0)
        
        return self.trans_function(net)


class NeuralNetwork(object):
    """
    Parameters
    ---------
    size:
        The number of perceptrons on each layer, excluding the input-layer.
    
    input_size:
        The number of inputs on the input-layer.
        
    E.g. size = [2, 1], and input_size = 2:
            
            o --- o
             \   /  \
               x     o --
             /   \  /
            o --- o
    """
    
    def __init__(self, size: List[int], input_size: int):
        
        for num in size:
            assert num > 0, "elements of `size` shall be positive."
            
        assert input_size > 0, "`input_size` shall be positive."
        
        self.size = size
        
        self.input_size = input_size
        
        self.layers = []
        
        for i in range(len(size)):
            if i == 0:
                l = [Perceptron(input_size) for j in range(size[i])]
            
            else:
                l = [Perceptron(size[i-1]) for j in range(size[i])]
            
            self.layers.append(l)
    
    
    def copy(self):
        
        c = deepcopy(self)
        
        return c

        
    def output(self, inputs: Array) -> Array:
        
        assert len(inputs) == self.input_size
                  
        x_inputs = inputs.copy()
        
        for layer in self.layers:
            x_outputs = np.zeros(len(layer))
            
            for i in range(len(layer)):
                perceptron = layer[i]
                x_outputs[i] = perceptron.output(x_inputs)
            
            x_inputs = x_outputs.copy()
        
        return x_outputs