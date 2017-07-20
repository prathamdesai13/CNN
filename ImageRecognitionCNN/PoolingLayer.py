"""
Pooling layer for CNN
"""
import numpy as np
from Layer import Layer


class PoolingLayer(Layer):

    def update_parameters(self, learning_rate):
        pass

    def __init__(self, pool_size, stride):

        self.input_vector = None
        self.N = None
        self.K = None
        self.W = None
        self.H = None
        self.window_width = pool_size[0]
        self.window_height = pool_size[1]
        self.S = stride
        self.output_width = None
        self.output_height = None
        self.max_indices = None

    def forwards_pass(self, input_vector, bool):

        self.input_vector = input_vector
        self.N = self.input_vector.shape[0]
        self.K = self.input_vector.shape[1]
        self.W = self.input_vector.shape[2]
        self.H = self.input_vector.shape[3]
        self.output_width = int((self.W - self.window_width) / self.S + 1)
        self.output_height = int((self.H - self.window_height) / self.S + 1)
        self.max_indices = np.zeros((self.N, self.K, self.output_width, self.output_height, 2))
        max_pool = np.zeros((self.N, self.K, self.output_width, self.output_height))

        for n in range(self.N):
            for filter in range(self.K):
                for h in range(self.output_height):
                    for w in range(self.output_width):
                        w_move = w * self.S
                        h_move = h * self.S

                        patch = self.input_vector[n, filter, w_move:w_move + self.window_width, h_move:h_move + self.window_height]
                        max_patch_value = np.max(patch)
                        location = list(zip(*np.where(max_patch_value == patch)))
                        location = (location[0][0] + self.S - 1, location[0][1] + self.S - 1)
                        self.max_indices[n, filter, w, h, 0] = location[0]
                        self.max_indices[n, filter, w, h, 1] = location[1]
                        max_pool[n, filter, w, h] = max_patch_value

        #print(max_pool.shape, "Max pool shape")
        return max_pool

    def backwards_pass(self, max_pool_gradient):

        input_vector_gradient = np.zeros(self.input_vector.shape)

        for n in range(self.N):
            for f in range(self.K):
                for h in range(self.output_height):
                    for w in range(self.output_width):

                        location = self.max_indices[n, f, w, h, :]
                        input_vector_gradient[n, f, int(location[0]), int(location[1])] = max_pool_gradient[n, f, w, h]

        return input_vector_gradient
