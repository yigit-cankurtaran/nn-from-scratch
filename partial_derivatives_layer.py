import numpy as np

# passed in gradients from next layer, example purposes
dvalues = np.array([[1.,1.,1.],
                   [2.,2.,2.],
                   [3.,3.,3.]])

# 3 sets of weights, 1 for each neuron, 4 inputs = 4 weights
weights = np.array([[0.2, 0.8, -0.5, 1],
                   [0.5, -0.91, 0.26, -0.5],
                   [-0.26, -0.27, 0.17, 0.87]]).T
# weights are kept transposed bc we want weights @ inputs to work OOTB

# sum weights of given input (dot product)
dinputs = np.dot(dvalues, weights.T) # have to re-transpose bc we need 3x3 @ 3x4
print(dinputs)

inputs = np.array([[1, 2, 3, 2.5],
[2., 5., -1., 2],
[-1.5, 2.7, 3.3, -0.8]]) # shape 3x4

# we want dweights to have the shape of weights
# weights are now (4,3) bc they're transposed
# (4,3) @ (3,3) = dweights.shape = (4,3) = weights.shape
dweights = np.dot(inputs.T, dvalues)
print(dweights)

biases = np.array([[2,3,0.5]])
# only need to sum dvalues bc bias derivative is 1
dbiases = np.sum(dvalues, axis=0, keepdims=True)

# forward pass
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0 # derivative of relu is 0 for stuff below 0 and 1 for above
