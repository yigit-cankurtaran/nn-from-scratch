import nnfs
import numpy as np
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weightregl1=0, weightregl2=0,
                 biasregl1=0, biasregl2=0):
        #initialize weights and biases
        # weights are random, biases are zero
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) #small, random magnitudes
        self.biases = np.zeros((1, n_neurons))
        self.weightregl1 = weightregl1
        self.weightregl2 = weightregl2
        self.biasregl1 = biasregl1
        self.biasregl2 = biasregl2

    def forward(self, inputs):
        # nothing new, turning previous stuff into a method
        self.inputs = inputs  # we want to keep track of inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        #parameter gradients
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #gradients on regularization
        # l1 on weights
        if self.weightregl1 > 0:
            # l1 is a sum operation,its derivative is 1 if weight>0 and -1 else
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weightregl1 * dL1
        #l2 on weights
        if self.weightregl2 > 0:
            # l2 squares and sums, derivative(x^2)=2x
            self.dweights += 2 * self.weightregl2 * self.weights

        #l1 on biases
        if self.biasregl1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.biasregl1 * dL1
        #l2 on biases
        if self.biasregl2 > 0:
            # l2 squares and sums, derivative(x^2)=2x
            self.dbiases += 2 * self.biasregl2 * self.biases

        # value gradient
        self.dinputs = np.dot(dvalues, self.weights.T)

class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1-rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

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

class Loss:
    def reg_loss(self, layer):
        reg_loss = 0 # default

        #l1 weight reg, only when bigger than 0
        if layer.weightregl1 > 0:
            reg_loss += layer.weightregl1 * np.sum(np.abs(layer.weights))

        #l2 weight reg, only when bigger than 0
        if layer.weightregl2 > 0:
            # if square here doesn't work just do weights * weights
            reg_loss += layer.weightregl2 * np.sum(layer.weights ** 2)

        #l1 bias reg, only when bigger than 0
        if layer.biasregl1 > 0:
            reg_loss += layer.biasregl1 * np.sum(np.abs(layer.biases))

        #l2 bias reg, only when bigger than 0
        if layer.biasregl2 > 0:
            # if square here doesn't work just do biass * biass
            reg_loss += layer.biasregl2 * np.sum(layer.biases ** 2)

        return reg_loss
        
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

class Optimizer_SGD():
    def __init__(self,learning_rate=1.0, decay=0.0, momentum=0.):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # before parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self,layer):
        if self.momentum:
            if not hasattr(layer, "weight_momentums"): # initially false
                # if layer doesn't have momentum (initially doesn't) create them w 0s
                layer.weight_momentums = np.zeros_like(layer.weights)
                # if no momentum for weights biases don't exist either, create them
                layer.bias_momentums = np.zeros_like(layer.biases)

            # take previous updates multiplied by retain factor, update w current gradients
            # gets dweights and dbiases during backprop, we pass layers into this
            weight_updates = self.momentum * layer.weight_momentums - self.current_lr * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_lr * layer.dbiases
            layer.bias_momentums = bias_updates
        else:    
            weight_updates += -self.current_lr * layer.dweights
            bias_updates += -self.current_lr * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1
    # in production optimizers the .step() method runs these 3 in order

class Optimizer_AdaGrad():
    def __init__(self,learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # before parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self,layer):
       if not hasattr(layer, "weight_cache"): # initially false
           # if layer doesn't have momentum (initially doesn't) create them w 0s
           layer.weight_cache = np.zeros_like(layer.weights)
           # if no momentum for weights biases don't exist either, create them
           layer.bias_cache = np.zeros_like(layer.biases)

       # update cache with squared current grad
       layer.weight_cache += layer.dweights**2
       layer.bias_cache += layer.dbiases**2

       #vanilla SGD parameter update + normalization with square rooted cache
       layer.weights += -self.current_lr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
       layer.biases += -self.current_lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
    # in production optimizers the .step() method runs these 3 in order

class Optimizer_RMSprop():
    def __init__(self,learning_rate=1.0, decay=0.0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # before parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self,layer):
       if not hasattr(layer, "weight_cache"): # initially false
           # if layer doesn't have momentum (initially doesn't) create them w 0s
           layer.weight_cache = np.zeros_like(layer.weights)
           # if no momentum for weights biases don't exist either, create them
           layer.bias_cache = np.zeros_like(layer.biases)

       # update cache with the RMSprop equation, different than AdaGrad
       layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights**2
       layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases**2

       # updates are implemented the same way as AdaProp
       layer.weights += -self.current_lr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
       layer.biases += -self.current_lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
    # in production optimizers the .step() method runs these 3 in order

class Optimizer_Adam:
    def __init__(self, learning_rate = 1e-3, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        # generally the default vals for Adam

        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay # optional, 0 by default
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
            if self.decay:
                self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
            if not hasattr(layer, 'weight_cache'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                layer.bias_cache = np.zeros_like(layer.biases)

            # update momentum
            layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1-self.beta_1) * layer.dweights
            layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1-self.beta_1) * layer.dbiases

            # get corrected momentum, iteration starts from 0 and we need to raise it to 1
            weight_momentums_corrected = layer.weight_momentums / (1-self.beta_1 ** (self.iterations+1))
            bias_momentums_corrected = layer.bias_momentums / (1-self.beta_1 ** (self.iterations+1))
            # update cache with squared current gradients
            layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2) * layer.dweights**2
            layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases**2
            #corrected cache
            weight_cache_corrected = layer.weight_cache / (1-self.beta_2 ** (self.iterations+1))
            bias_cache_corrected = layer.bias_cache / (1-self.beta_2 ** (self.iterations+1))

            # sgd with normalization
            layer.weights += -self.current_lr * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
            layer.biases += -self.current_lr * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
            self.iterations += 1

X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 512, weightregl2=5e-4, biasregl2=5e-4) # 2 inputs 64 outputs, l2 reg
# we usually add regularization terms to the hidden layers only

dense2 = Layer_Dense(512, 3) # 64 inputs 3 outputs
activation1 = Activation_ReLU() # creating relu object
loss_activation = Activation_Softmax_Loss_CCE() # will replace the separate loss and activation
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5) # higher LR, added LR decay
dropout1 =Layer_Dropout(0.1)

for epoch in range(10001):
    dense1.forward(X) # forward pass of training data
    activation1.forward(dense1.output) # forward pass through relu
    dropout1.forward(activation1.output) # adding dropout after the first layer
    dense2.forward(dropout1.output) # dense2 gets relu's output as input
    data_loss = loss_activation.forward(dense2.output, y) # both softmax and the loss
    # loss becomes what the forward method returns
    # implementing regularization here
    reg_loss = loss_activation.loss.reg_loss(dense1) + loss_activation.loss.reg_loss(dense2)
    loss = data_loss + reg_loss
    
    if len(y.shape) == 2: # convert from one hot matrix
        y = np.argmax(y, axis=1)
    
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)
    # predictions == y creates a boolean array, how many 1s / total bools

    if not epoch % 100: # same thing as epoch % 100 == 0
        print(f"epoch:{epoch}\nacc:{accuracy}\ndata loss:{data_loss}\nregularization loss:{reg_loss}\nlr:{optimizer.current_lr}")

    #backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs) # adding dropout right before the activation
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)
    
    # after we get the gradients we update the network layer parameters
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# validating the model
X_test, y_test = spiral_data(samples=1000, classes=3) # test dataset
print("test dataset created")

# forward pass
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
data_loss = loss_activation.forward(dense2.output, y_test)
reg_loss = loss_activation.loss.reg_loss(dense1) + loss_activation.loss.reg_loss(dense2)
loss = data_loss + reg_loss

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)

print(f"validation acc:{accuracy}, loss:{loss}")
