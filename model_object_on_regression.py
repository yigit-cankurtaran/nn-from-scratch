from nnfs.datasets import sine_data
from dense_layer_class import Layer_Dense, Activation_ReLU, Optimizer_Adam
from regression_model import Activation_Linear, Loss_MeanSquaredError

X, y = sine_data()


class Model:
    def __init__(self):
        # list of network objects
        self.layers = []

    def add(self, layer):
        # adding objects to the model
        self.layers.append(layer)

    def set(self, *, loss, optim):
        """
        to set loss function and optimizer
        \* here means that anything after it are keywords, passed by names and vals
        """
        self.loss = loss
        self.optim = optim


model = Model()  # instantiate object

# adding layers
model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

model.set(
    loss=Loss_MeanSquaredError(), optim=Optimizer_Adam(learning_rate=0.005, decay=1e-3)
)

print(model.layers)
