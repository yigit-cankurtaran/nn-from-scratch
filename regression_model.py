import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import sine_data
from dense_layer_class import Activation_ReLU, Layer_Dense, Loss, Optimizer_Adam

nnfs.init()

X, y = sine_data()  # using this as data
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


dense1 = Layer_Dense(
    1, 64, weightregl2=5e-4, biasregl2=5e-4
)  # imported from dense_layer_class
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 1)  # output
activation2 = Activation_Linear()
loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)

# using standard deviation with a random constant
# so that the accuracy we use can adapt to the inputs at hand
# e.g. if y is house prices in dollars (σ ≈ 100 k) the tolerance becomes 400 $.
# if y is temperature in °C (σ ≈ 10 °C) the tolerance becomes 0.04 °C.
acc_precision = np.std(y) / 250

for epoch in range(10001):
    # forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    data_loss = loss_function.calculate(activation2.output, y)
    reg_loss = loss_function.reg_loss(dense1) + loss_function.reg_loss(dense2)
    loss = data_loss + reg_loss

    predictions = activation2.output
    accuracy = np.mean(np.absolute(predictions - y) < acc_precision)

    if not epoch % 100:  # same thing as epoch % 100 == 0
        print(
            f"epoch:{epoch}\nacc:{accuracy}\ndata loss:{data_loss}\
            \nregularization loss:{reg_loss}\nlr:{optimizer.current_lr}"
        )

    # backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # weight and bias update
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# visualizing
X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

plt.plot(X_test, y_test)  # normal data
plt.plot(X_test, activation2.output)  # normal data
plt.show()
