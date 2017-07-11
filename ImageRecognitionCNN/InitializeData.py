"""
Initialize mnist data and split for training and validation
"""
import numpy as np
import sklearn.datasets as sk


def initialize():

    data = sk.fetch_mldata('MNIST original', data_home='./MNIST')

    N = 60000
    inputs = np.reshape(data.data[:N], (-1, 1, 28, 28)) / 255.0
    labels = data.target[:N]
    #print(labels.shape, "shape")
    validation_inputs = np.reshape(data.data[N:], (-1, 1, 28, 28)) / 255.0
    validations_labels = data.target[N:]

    num_classes = np.unique(labels).shape[0]

    inputs, labels = randomize(inputs, labels)
    validation_inputs, validations_labels = randomize(validation_inputs, validations_labels)

    return inputs, labels, validation_inputs, validations_labels, num_classes

def randomize(x, y):

    N = x.shape[0]
    random_indices = np.random.random_integers(0, N - 1, N)
    x = x[random_indices, ...]
    y = y[random_indices, ...]

    return x, y

