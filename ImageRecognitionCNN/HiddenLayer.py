"""
Hidden layer in neural network
"""
from Layer import Layer
import numpy as np


class HiddenLayer(Layer):

    def __init__(self, num_classes, num_inputs):

        self.input = None
        self.weights = 0.0001 * np.random.rand(num_classes, num_inputs)
        self.bias = np.random.rand(1, num_classes)
        self.velocity = np.zeros(self.weights.shape)
        self.weights_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.bias.shape)
        self.momentum = 0.9

    def forwards_pass(self, input_vector, bool):

        self.input = input_vector

        weighted_sum = np.dot(self.input, self.weights.T) + self.bias

        return weighted_sum

    def backwards_pass(self, output_gradient):

        N = self.input.shape[0]
        self.weights_gradient = np.dot(self.input.T, output_gradient).T / N
        self.bias_gradient = np.sum(output_gradient, axis=0)
        input_gradient = np.dot(output_gradient, self.weights)

        return input_gradient

    def update_parameters(self, learning_rate):

        self.velocity = np.multiply(self.momentum, self.velocity) - (learning_rate * self.weights_gradient)
        self.weights = self.weights + self.velocity

        self.bias = self.bias - (learning_rate * self.bias_gradient)
