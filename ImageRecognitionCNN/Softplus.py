"""
Softplus activations function
"""
import numpy as np
from Layer import Layer


class Softplus(Layer):

    def __init__(self):

        self.input = None
        
    def forwards_pass(self, input_vector):
        self.input = input_vector
        unit = np.log(1 + np.exp(self.input))
        return unit

    def backwards_pass(self, output_gradient):

        input_gradient = output_gradient * self.softplus_gradient(self.input)

        return input_gradient

    def update_parameters(self, learning_rate):
        pass

    def softplus_gradient(self, input):

        dx = 1.0 / (1.0 + np.exp(-input))

        return dx
