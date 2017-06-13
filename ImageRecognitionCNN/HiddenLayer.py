"""
Hidden layer in neural network
"""
from Layer import Layer
import numpy as np


class HiddenLayer(Layer):

    def __init__(self, num_classes, num_inputs):

        self.input = None
        self.weights = np.random.normal(scale=0.1, size=(num_inputs, num_classes))
        self.bias = np.random.normal(scale=0.1, size=num_classes)
        self.weights_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.bias.shape)

    def forwards_pass(self, input_vector):

        self.input = input_vector

        weighted_sum = np.dot(self.input, self.weights) + self.bias

        return weighted_sum

    def backwards_pass(self, output_gradient):

        N = output_gradient.shape[0]
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        self.weights_gradient /= N
        self.weights_gradient -= 0.02 * self.weights
        self.bias_gradient = np.mean(output_gradient, axis=0)
        input_gradient = np.dot(output_gradient, self.weights.T)

        return input_gradient

    def update_parameters(self, learning_rate):

        self.weights -= self.weights_gradient * learning_rate
        self.bias -= self.bias_gradient * learning_rate


