"""
Sigmoid activation function
"""
import scipy.special as spl
import numpy as np
from Layer import Layer

class Sigmoid(Layer):

    def __init__(self):

        self.input_vector = None

    def forwards_pass(self, input_vector, bool):
        self.input_vector = input_vector

        unit = spl.expit(input_vector)

        return unit


    def update_parameters(self, learning_rate):
        pass

    def backwards_pass(self, output_gradient):

        return np.multiply(output_gradient, self.sigmoid_derivative())

    def sigmoid_derivative(self):

        return (1 - self.forwards_pass(self.input_vector, True))*self.forwards_pass(self.input_vector, True)
