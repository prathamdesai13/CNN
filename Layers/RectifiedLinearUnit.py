"""
ReLU for convolutions in CNN
"""
import numpy as np
from Layer import Layer


class ReLU(Layer):

    def __init__(self):

        self.input = None

    def forwards_pass(self, input_vector, bool):
        self.input = input_vector

        unit = np.maximum(0.0, self.input)

        return unit

    def backwards_pass(self, output_gradient):

        input_gradient = output_gradient * self.ReLU_gradient(self.input)

        return input_gradient

    def update_parameters(self, learning_rate):
        pass

    def ReLU_gradient(self, input):

        dx = np.zeros(input.shape)

        dx[input >= 0] = 1

        return dx
