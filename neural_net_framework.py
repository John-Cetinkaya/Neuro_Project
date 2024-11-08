"""This module is a extremely basic implementation of a neural network to allow for experimentation
with neuro evolution"""

import numpy as np

class LayerDense:
    """Creates a single layer of the neural network"""
    def __init__(self, n_inputs, n_neurons, activation = None):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.uniform(-10,10, (1,n_neurons))
        self.output = None
        #np.zeros((1,n_neurons))
        self.activation = activation

    def forward(self, inputs):
        """Calculates the output of the layer to allow for the next layer to receive as inputs"""
        if self.activation == "ReLU":
            layer_output = np.dot(inputs, self.weights) + self.biases
            activation = ActivationReLU()
            activation.forward(layer_output)
            self.output = activation.output
        if self.activation == "Softmax":
            layer_output = np.dot(inputs, self.weights) + self.biases
            activation = ActivationSoftmax()
            activation.forward(inputs)
            self.output = activation.output
        else:
            self.output = np.dot(inputs, self.weights) + self.biases

    def set_weights(self, weights):
        """Manually sets weights for the layer. Does not check shape so must be careful when creating genomes"""
        self.weights = weights

    def set_biases(self, biases):
        """unused but manually sets biases"""
        self.biases = biases


class ActivationReLU:
    """Creates the Rectified liner unit activation function"""
    def forward(self, inputs):
        """calculates output"""
        self.output = np.maximum(0, inputs)

class ActivationSoftmax:
    """Creates softmax activation function"""
    def forward(self, inputs):
        """Calculates output"""
        exp_values = np.exp(inputs- np.max(inputs, axis= 1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class NNModel:
    """Takes a list of layers as an input and creates a model that can be
    ran or edited"""
    def __init__(self, layers:list, inputs = None):
        self.layers = layers
        self.inputs = inputs

    def predict(self):
        """Calculates a output given that an input has already been assigned with the correct shape"""
        holding_inputs = self.inputs
        for layer in self.layers:
            layer.forward(holding_inputs)
            output = layer.output
            holding_inputs = output
        return output

    def set_inputs(self, inputs):
        """Sets inputs to the neural network"""
        self.inputs = inputs

    def set_weights(self, genome):
        """Manually sets weights for neural network, does not check shape so be mindful when setting weights"""
        for i, layer in enumerate(self.layers):
            layer.set_weights(genome[i])

    def set_biases(self, genome):
        """unused but same as weights but with biases"""
        for i, layer in enumerate(self.layers):
            layer.set_biases(genome[i])



if __name__ == "__main__":
    X = [50,400,721,90]
    layers = [LayerDense(4, 8, activation= "ReLU"),
            LayerDense(8, 8, activation= "ReLU"),
            LayerDense(8, 8, activation= "ReLU"),
            LayerDense(8, 4, activation= "ReLU"),
            LayerDense(4, 4, activation="Softmax")]

    model = NNModel(layers, X)
    print(model.predict())