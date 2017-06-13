"""
Final layer of neural network
"""
from __future__ import division

from Layer import Layer
import numpy as np


class FinalLayer(Layer):
    """softmax with cross entropy loss function"""

    def __init__(self):
        self.input_vector = None
        self.scores = None

    def update_parameters(self, learning_rate):
        pass

    def forwards_pass(self, input_vector):
        self.input_vector = input_vector

        exp = np.exp(input_vector - np.amax(input_vector, axis=1, keepdims=True))
        sum_exp = np.sum(exp, axis=1, keepdims=True)
        norm_exp = (exp / sum_exp)
        self.scores = norm_exp

        return norm_exp

    def backwards_pass(self, output_gradient):
        pass

    def gradient(self, output_vector_prediction, output_vector_label):

        gradient = output_vector_prediction - output_vector_label

        return gradient

    def loss(self, output_vector_prediction, output_vector_label):

        epsilon = 1e-15
        N = output_vector_label.shape[0]
        output_vector_prediction = np.clip(output_vector_prediction, epsilon, 1 - epsilon)
        output_vector_prediction /= output_vector_prediction.sum(axis=1, keepdims=True)
        soft_loss = -np.sum(output_vector_label * np.log(output_vector_prediction))
        norm_soft_loss = soft_loss / N

        return norm_soft_loss

