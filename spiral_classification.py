import numpy as np
from nnfs.datasets import spiral_data
from dense_layer_class import (
    Layer_Dense,
    Activation_ReLU,
    Activation_Sigmoid,
    Loss_BinaryCrossEntropy,
    Optimizer_Adam,
    Layer_Dropout
)

# for binary cross entropy test
X, y = spiral_data(samples=100, classes=2)  # changing from 3 classes to 2
y = y.reshape(-1, 1)  # reshape to be a list of lists
# inner list contains 1 output per each output neuron

dense1 = Layer_Dense(
    2, 64, weightregl2=5e-4, biasregl2=5e-4
)  # 2 inputs 64 outputs, l2 reg
# we usually add regularization terms to the hidden layers only

dense2 = Layer_Dense(64, 1)  # 64 inputs 1 output
# went from 3 outputs to 1 output bc the binary classification

activation1 = Activation_ReLU()  # creating relu object
activation2 = Activation_Sigmoid()

loss_function = Loss_BinaryCrossEntropy()

optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)  # higher LR, added LR decay
dropout1 = Layer_Dropout(0.1)

for epoch in range(10001):
    dense1.forward(X)  # forward pass of training data
    activation1.forward(dense1.output)  # forward pass through relu
    dropout1.forward(activation1.output)  # adding dropout after the first layer
    dense2.forward(dropout1.output)  # dense2 gets relu's output as input
    # data_loss = loss_activation.forward(dense2.output, y)  # both softmax and the loss
    # loss becomes what the forward method returns
    # implementing regularization here
    activation2.forward(dense2.output)
    data_loss = loss_function.calculate(activation2.output, y)

    reg_loss = loss_function.reg_loss(dense1) + loss_function.reg_loss(dense2)
    loss = data_loss + reg_loss

    # for the multi class thing
    # if len(y.shape) == 2:  # convert from one hot matrix
    #     y = np.argmax(y, axis=1)

    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)
    # predictions == y creates a boolean array, how many 1s / total bools

    if not epoch % 100:  # same thing as epoch % 100 == 0
        print(
            f"epoch:{epoch}\nacc:{accuracy}\ndata loss:{data_loss}\nregularization loss:{reg_loss}\nlr:{optimizer.current_lr}"
        )

    # backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    dropout1.backward(dense2.dinputs)  # adding dropout right before the activation
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    # after we get the gradients we update the network layer parameters
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# validating the model
X_test, y_test = spiral_data(samples=1000, classes=2)  # test dataset
# reshape this y_test as well
y_test = y_test.reshape(-1, 1)
print("test dataset created")

# forward pass (no dropout during validation)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.calculate(activation2.output, y_test)
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)

print(f"validation acc:{accuracy}, loss:{loss}")