from nnfs.datasets import sine_data

X, y = sine_data()


class Model:
    def __init__(self):
        # list of network objects
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
