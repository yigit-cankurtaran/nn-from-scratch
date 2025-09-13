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
class Loss_MeanSquaredError(Loss):  # a.k.a. L2 loss
    def forward(self, y_pred, y_true):
        # (target - prediction)^2
        return np.mean((y_true - y_pred) ** 2, axis=-1)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # num of outputs every sample, using the first sample to count them
        outputs = len(dvalues[0])

        # grad on values

        # derivative = -2/j * (y - yhat) j = samples
        # we divide by outputs here because we're taking the derivative of a mean
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # normalize grad
        self.dinputs /= samples


class Loss_MeanAbsoluteError(Loss):  # a.k.a. L1 loss
    def forward(self, y_pred, y_true):
        return np.mean(np.abs(y_true - y_pred), axis=-1)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
