import nnfs
import numpy as np

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
        #calculate output values from inputs, weights and biases
        pass
        
Layer_Dense(2, 4)
