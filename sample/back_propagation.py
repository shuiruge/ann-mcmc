"""
Description
---------
Module of construction of ANN. It's written by numpy.
"""

import numpy as np
from random import gauss, uniform
from copy import deepcopy
import ann


# Typing Hint
# ----------------------
from typing import List, Mapping, Tuple

Array = np.array(List[float])
Sample = Array
Samples = List[List[float]] # It should List[Array]!
# ----------------------




    
def delta(net: ann.NeuralNetwork,
           sample: Sample,
           error_function_deriv: List[Mapping[Tuple(Sample,
                                                    ann.NeuralNetwork
                                                    ),
                                              float]
                                      ],
           learning_rate=0.1
           ) -> List[List[float]]:
    """ Compute the deltae of ...
    """
    
    assert len(sample) == net.layers[0].input_size
    
    result = []
    
    # compute output for each perceptron, and order then into a list of layers
    # of perceptrons.
    outputs = []
    layer_inputs = sample
    
    for layer in net.layers:
        
        layer_outputs = layer.output(layer_inputs)
        outputs.append(layer_outputs)
        
        layer_inputs = layer_outputs.copy()
    
    outputs = np.array(outputs)
    
    for i in range(len(net.layers)):
        
        j = net.size - 1 - i
        
        if j == net.size - 1:
            deltae = outputs[j] * (1 - outputs[j]) * deriv(output[-1])[j]
            result.append(deltae)
        
        else:
            weighted = 
        
    
    return result_net
    
    
def train(net: ann.NeuralNetwork,
          samples: Samples,
          error_function: Mapping[Tuple(Sample, ann.NeuralNetwork), float],
          goal,
          learning_rate=0.1,
          epoches=10
          ) -> ann.NeuralNetwork:
    """ Back propagation trainer.
    
    Parameters
    ---------
    goal:
        
    
    epoches:
        
    """
    
    result_net = net.copy()
    
    for epoch in range(epoches):
        
        for sample in samples:
            
            reach_the_goal = abs(error_function(sample, result_net)) < goal
            
            if reach_the_goal:
                return result_net
            
            else:
                result_net = update(result_net, sample,
                                    error_function, learning_rate
                                    )
    return result_net
    
    