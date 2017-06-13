"""
Initialize mnist data and split for training and validation
"""
import numpy as np
import sklearn.datasets as sk


def initialize():

    data = sk.fetch_mldata('MNIST original', data_home='./MNIST')

    split = 60000
    inputs = np.reshape(data.data[:split], (-1, 1, 28, 28))/255.0
    labels = data.target[:split]
    validation_inputs = np.reshape(data.data[split:], (-1, 1, 28, 28))/255.0
    validations_labels = data.target[split:]
    num_classes = np.unique(labels).shape[0]

    return inputs, labels, validation_inputs, validations_labels, num_classes


