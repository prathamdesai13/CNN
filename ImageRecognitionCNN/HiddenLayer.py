"""
Hidden layer in neural network
"""
from Layer import Layer
import numpy as np


class HiddenLayer(Layer):

    def __init__(self, num_classes, num_inputs):

        self.input = None
        self.weights = np.zeros((num_classes, num_inputs))
        self.weights_gradient = np.zeros(self.weights.shape)

    def forwards_pass(self, input_vector):

        self.input = input_vector

        weighted_sum = np.dot(self.input, self.weights.T)

        return weighted_sum

    def backwards_pass(self, output_gradient):
        N = self.input.shape[0]
        self.weights_gradient = -np.dot(self.input.T, output_gradient)
        self.weights_gradient = self.weights_gradient / N
        self.weights_gradient = self.weights_gradient.T
        input_gradient = np.dot(output_gradient, self.weights)

        return input_gradient

    def update_parameters(self, learning_rate):

        self.weights = self.weights - (learning_rate * self.weights_gradient)


