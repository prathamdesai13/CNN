"""
Reformatting pools to fit into hidden layer
"""

import numpy as np
from Layer import Layer


class Vectorize(Layer):

    def __init__(self):
        self.input = None

    def forwards_pass(self, input_vector):
        self.input = input_vector
        reshaped_input_vector = np.reshape(input_vector, (input_vector.shape[0], -1))

        return reshaped_input_vector

    def backwards_pass(self, output_gradient):
        output_gradient = np.reshape(output_gradient, self.input.shape)
        return output_gradient

    def update_parameters(self, learning_rate):
        pass

