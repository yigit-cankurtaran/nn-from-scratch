import nnfs
import numpy as np
from nnfs.datasets import spiral_data

nnfs.init()

# print(np.random.randn(2,6)) # means shape of (2,6)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #initialize weights and biases
        # weights are random, biases are zero
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) #small, random magnitudes
        self.biases = np.zeros((1, n_neurons))
        print(f"weights are {self.weights}")
        print(f"biases are {self.biases}")
    def forward(self, inputs):
        # nothing new, turning previous stuff into a method
        self.output = np.dot(inputs, self.weights) + self.biases
        pass
    # we'll update weights and biases and such with a backward pass later

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        # unnormalized probabilities
        # we subtract the largest input from every neuron, it can wreck neurons
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalize
        # axis=1 bc we want the sum of rows
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
dense2 = Layer_Dense(3, 3) # 3 rows 3 columns
activation1 = Activation_ReLU() # creating relu object
activation2 = Activation_Softmax() # creating softmax object

dense1.forward(X) # forward pass of training data
activation1.forward(dense1.output) # forward pass through relu
dense2.forward(activation1.output) # dense2 gets relu's output as input
activation2.forward(dense2.output) # softmax forward pass

# result after 2 layers + softmax for output

print(f"output of first few samples after softmax:\n {activation2.output[:5]}")
