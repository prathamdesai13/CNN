"""
ReLU for convolutions in CNN
"""
import numpy as np
from Layer import Layer


class ReLU(Layer):

    def __init__(self, type):

        self.input = None
        self.type = type


    def forwards_pass(self, input_vector):
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
        if self.type == 1:
            for n in range(input.shape[0]):
                for f in range(input.shape[1]):
                    for h in range(input.shape[3]):
                        for w in range(input.shape[2]):

                            if input[n, f, w, h] > 0:
                                dx[n, f, w, h] = int(1.0)
                            else:
                                dx[n, f, w, h] = int(0.0)
        elif self.type == 2:

            dx[input > 0] = 1


        return dx
