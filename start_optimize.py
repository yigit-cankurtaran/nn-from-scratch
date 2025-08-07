import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from dense_layer_class import Layer_Dense, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossEntropy
matplotlib.use("Agg") # to not open a window

import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

# generating a random set of weights, keeping it every time we have a better result

# X is a 2D array containing the features for our dataset
# y is a 1d array containing class labels for each sample
X, y = vertical_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3) # 2 inputs
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3) # 3 inputs bc last output was 3
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()

lowest_loss = 99999 #random big number
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
# copy ensures a full copy instead of a reference to the object

epoch = 10000

for iter in range(epoch):
    # iterate on the set of weights and biases for iteration
    # choosing weights randomly takes up way too much time
    dense1.weights += 0.05 * np.random.randn(2,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense2.biases += 0.05 * np.random.randn(1,3)

    # training data forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # loss calculation
    loss = loss_function.calculate(activation2.output, y)

    # accuracy calc
    predictions = np.argmax(activation2.output, axis=1)
    acc = np.mean(predictions==y)

    if loss < lowest_loss:
        print(f"new best found on iteration {iter}, loss={loss}, accuracy={acc}")
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        # if current iteration doesn't improve the loss, revert back to usual
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap="brg")
plt.savefig("vertical_data.png") # saving as vertical_data.png
