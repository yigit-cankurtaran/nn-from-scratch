import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data

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
