import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import sine_data
from dense_layer_class import Loss

nnfs.init()

X, y = sine_data()
plt.plot(X, y)
plt.show()


# regression models use linear activation for output
class Activation_Linear:
    def forward(self, inputs):
        # x = y
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        # 1 * dvalues = dvalues
        self.dinputs = dvalues.copy()


# imported Loss class from the previous example we used, inherited here
class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2, axis=-1)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # num of outputs every sample, using the first sample to count them
        outputs = len(dvalues[0])

        # grad on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # normalize grad
        self.dinputs /= samples
