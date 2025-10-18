from nnfs.datasets import sine_data
from dense_layer_class import Layer_Dense, Activation_ReLU


class Activation_Linear:
    def forward(self, inputs):
        # linear activation simply passes values through
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        # derivative wrt inputs is 1 so gradient flows unchanged
        self.dinputs = dvalues.copy()


X, y = sine_data()


class Model:
    def __init__(self):
        # list of network objects
        self.layers = []

    def add(self, layer):
        # adding objects to the model
        self.layers.append(layer)


model = Model()  # instantiate object

# adding layers
model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

print(model.layers)
