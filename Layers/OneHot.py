"""
One hot transformation for labels, where n-th digit is represented with n-th dimension having 1
"""
import numpy as np


def one_hot(labels):

    classes = np.unique(labels)
    N = classes.shape[0]
    num_labels = labels.shape[0]
    one_hot_labels = np.zeros(labels.shape + (N,))
    for j in range(N):
        for i in range(num_labels):
            if labels[i] == classes[j]:
                one_hot_labels[i, j] = 1
            else:
                one_hot_labels[i, j] = 0

    return one_hot_labels


def reverse(one_hot_labels):

    return np.argmax(one_hot_labels, axis=-1)



