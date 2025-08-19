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
        self.inputs = inputs  # we want to keep track of inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        #parameter gradients
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # value gradient
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs # we want to keep track of inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # we'll modify the vars so we make a copy
        self.dinputs = dvalues.copy()
        # if input vals negative zero
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        # unnormalized probabilities
        # we subtract the largest input from every neuron, it can wreck neurons
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalize
        # axis=1 bc we want the sum of rows
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        #outputs and gradients

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output = single_output.reshape(-1,1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #samplewise gradient
            self.dinputs[index] = np.dot(jacobian, single_dvalues)

#later we'll add more code to this
class Loss:
    def calculate(self, output, y):
        # sample losses
        sample_losses = self.forward(output, y)

        #mean loss
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss): #inherits Loss class
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # clip data to prevent numerical issues
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, y_true, dvalues):
        samples = len(dvalues)
        labels = len(dvalues[0]) # using first sample to count labels

        # if sparse turn into one hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # derivative of CCE
        self.dinputs = -y_true / dvalues
        #normalize
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CCE():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        """
            dvalues = softmax probabilities from forward pass
            
        """
        samples = len(dvalues)
        # if labels are one hot encoded turn them discrete
        # discrete = just the index of the correct class
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # copy for safe mods
        self.dinputs = dvalues.copy()
        # calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # take the value of the true class, subtract 1 from it
        # wrong classes stay as 0.
        # e.g. confidence in the correct output is 0.7, gradient becomes -0.3

        # normalize gradient
        self.dinputs = self.dinputs / samples


X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
dense2 = Layer_Dense(3, 3) # 3 rows 3 columns
activation1 = Activation_ReLU() # creating relu object
activation2 = Activation_Softmax() # creating softmax object
loss_function = Loss_CategoricalCrossEntropy()

dense1.forward(X) # forward pass of training data
activation1.forward(dense1.output) # forward pass through relu
dense2.forward(activation1.output) # dense2 gets relu's output as input
activation2.forward(dense2.output) # softmax forward pass

# result after 2 layers + softmax for output
print(f"output of first few samples after softmax:\n {activation2.output[:5]}")

# forward pass through loss function
# softmaxed the output layer, we're passing in the output of that to the loss function
loss = loss_function.calculate(activation2.output, y)
print(f"loss: {loss}")

if len(y.shape) == 2: # convert from one hot matrix
    y = np.argmax(y, axis=1)

predictions = np.argmax(activation2.output, axis=1)
accuracy = np.mean(predictions == y)
# predictions == y creates a boolean array, how many 1s / total bools

print(f"accuracy: {accuracy}")
