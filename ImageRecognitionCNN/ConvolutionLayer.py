"""
Convolution layer for CNN
"""
import numpy as np

from Layer import Layer


class ConvolutionLayer(Layer):

    def __init__(self, num_kernels, kernel_size, stride):
        self.input_vector = None
        self.N = None
        self.D = None
        self.W = None
        self.H = None
        self.kernels = None
        self.kernels_gradient = None
        self.width_out = None
        self.height_out = None
        self.feature_maps = None
        self.F = kernel_size

        self.K = num_kernels
        self.S = stride

    def forwards_pass(self, input_vector, bool):
        self.input_vector = input_vector
        self.N = input_vector.shape[0]
        self.D = input_vector.shape[1]
        self.W = input_vector.shape[2]
        self.H = input_vector.shape[3]
        if bool == True:
            self.kernels = np.random.random((self.D, self.K, self.F, self.F))

        self.width_out = int((self.W - self.F) / self.S + 1)
        self.height_out = int((self.H - self.F) / self.S + 1)

        self.feature_maps = np.zeros((self.N, self.K, self.width_out, self.height_out))

        for n in range(self.N):
            for filter in range(self.K):
                for h in range(self.height_out):
                    for w in range(self.width_out):
                            w_move = self.S * w
                            h_move = self.S * h
                            for channel in range(self.D):
                                image_patch = self.input_vector[n, channel, w_move:w_move + self.F, h_move:h_move + self.F]
                            #chejc if this is correct
                                self.feature_maps[n, filter, w, h] = np.sum(image_patch * self.kernels[channel, filter, :, :])

        return self.feature_maps



    def backwards_pass(self, feature_maps_gradient):
        N = self.input_vector.shape[0]
        input_vector_gradient = np.zeros(self.input_vector.shape)
        self.kernels_gradient = np.zeros(self.kernels.shape)

        for n in range(self.N):
            for filter in range(self.K):
                for h in range(0, self.height_out, self.S):
                    for w in range(0, self.width_out, self.S):
                            input_vector_gradient[n, :, w:w + self.F, h:h + self.F] = feature_maps_gradient[n, filter, w, h] * self.kernels[:, filter, :, :]

                for h in range(feature_maps_gradient.shape[3]):
                    for w in range(feature_maps_gradient.shape[2]):

                        self.kernels_gradient[:, filter, :, :] = feature_maps_gradient[n, filter, w, h] * self.input_vector[n, :, w*self.S:w*self.S + self.F, h*self.S:h*self.S + self.F]

        self.kernels_gradient[...] /= N

        return input_vector_gradient

    def update_parameters(self, learning_rate):

        self.kernels -= self.kernels_gradient * learning_rate


