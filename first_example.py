import numpy as np
import matplotlib as mpl

#modeling a single neuron

inputs = [1,2,3]
weights = [0.2, 0.8, -0.5]
# weights are respective to each input
bias = 2
# 1 neuron = 1 bias
# numbers are random

output = 0
for i in range(len(inputs)):
    output += inputs[i] * weights[i]
    # output = activation((weights * inputs) + bias)
    # no activation yet

output += bias
print(output) #prints 2.3
