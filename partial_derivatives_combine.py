import numpy as np

# passed in gradient from next layer
dvalues = np.array([[1.,1.,1.],
                   [2.,2.,2.],
                   [3.,3.,3.]])

# 3x4 inputs
inputs = np.array([[1,2,3,2.5],
                  [2.,5.,-1.,2],
                  [-1.5,2.7,3.3,-0.8]])

# don't forget to keep weights transposed!
weights = np.array([[0.2,0.8,-0.5,1],
                   [0.5,-0.91,0.26,-0.5],
                   [-0.26, -0.27, 0.17, 0.87]]).T

# one bias for each neuron
biases = np.array([[2,3,0.5]])

#forward pass
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

#testing backprop
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

# dense layer
dinput = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, drelu) # we want dweights to be the shape of weights.T
dbiases = np.sum(drelu, axis=0, keepdims=True)

# decreasing loss by tuning weights and biases
# derivative = direction and magnitude of change
# since we want to decrease the loss, we move in the OPPOSITE direction
# 0.001 is the learning rate
weights += -0.001 * dweights
biases += -0.001 * dbiases
print(weights)
print(biases)
