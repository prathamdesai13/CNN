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
        self.error = None

    def update_parameters(self, learning_rate):
        pass

    def forwards_pass(self, input_vector):

        input_vector = input_vector - np.max(input_vector)
        norm_exp = (np.exp(input_vector).T / np.sum(np.exp(input_vector), axis=1)).T
        epsilon = 1e-3
        norm_exp = np.clip(norm_exp, epsilon, 1 - epsilon)
        return norm_exp

    def backwards_pass(self, output_gradient):

        gradient = output_gradient * self.error

        return gradient

    def gradient(self, output_vector_prediction, output_vector_label):

        self.error = output_vector_label - output_vector_prediction

    def loss(self, output_vector_prediction, output_vector_label):

        N = output_vector_label.shape[0]
        soft_loss = -np.sum(output_vector_label * np.log(output_vector_prediction))
        norm_soft_loss = soft_loss / N

        return norm_soft_loss

