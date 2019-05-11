"""
Final layer of neural network
"""

from Layer import Layer
import numpy as np
import scipy.special as spl


class FinalLayer(Layer):
    """softmax with cross entropy loss function"""

    def __init__(self):

        self.error = None
        self.input_vector = None

    def update_parameters(self, learning_rate):
        pass

    def forwards_pass(self, input_vector, bool):
        self.input_vector = input_vector
        norm_exp = self.softmax(self.input_vector)

        return norm_exp

    def backwards_pass(self, output_gradient):

        gradient = output_gradient * self.error

        return gradient

    def gradient(self, output_vector_prediction, output_vector_label):

        self.error = output_vector_prediction - output_vector_label


    def loss(self, output_vector_prediction, output_vector_label):
        # adjust probabilites by 1e-15 and normalize for numerical stability
        output_vector_prediction = np.clip(output_vector_prediction, 1e-15, 1 - 1e-15)
        norm_pred = output_vector_prediction / output_vector_prediction.sum(axis=1, keepdims=True)

        N = output_vector_label.shape[0]
        soft_loss = -np.sum(np.nan_to_num(output_vector_label * np.log(norm_pred)))
        norm_soft_loss = soft_loss / N

        return norm_soft_loss

    def softmax(self, x):

        x -= np.max(x, axis=1, keepdims=True)
        unit = np.exp(x)
        unit /= np.sum(np.exp(x), axis=1, keepdims=True)

        return unit

    def sigmoid(self, x):

        unit = spl.expit(x)

        return unit

    def sigmoid_derivative(self, x):

        return (1 - self.sigmoid(x)) * self.sigmoid(x)

