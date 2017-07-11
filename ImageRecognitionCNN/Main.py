"""
Main py file
"""
from NeuralNetwork import NeuralNetwork
from FinalLayer import FinalLayer
from HiddenLayer import HiddenLayer
from InitializeData import initialize
from ConvolutionLayer import ConvolutionLayer
from RectifiedLinearUnit import ReLU
from Vectorize import Vectorize
from PoolingLayer import PoolingLayer
import OneHot as onehot
import time
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np

def run(learning_rate, iterations, split_size, downsample_size):

    inputs, labels, validation_inputs, validation_labels, num_classes = initialize()
    labels = onehot.one_hot(labels)
    validation_labels = onehot.one_hot(validation_labels)

    reluLayer1 = ReLU()
    reluLayer2 = ReLU()

    vectorizeLayer = Vectorize()
    hiddenLayer = HiddenLayer(150, 784)
    hiddenLayer2 = HiddenLayer(75, 150)
    hiddenLayer3 = HiddenLayer(10, 75)
    finalLayer = FinalLayer()

    LAYERS = [vectorizeLayer, hiddenLayer, reluLayer1, hiddenLayer2, reluLayer2, hiddenLayer3, finalLayer]

    t1 = time.time()

    NEURALNETWORK = NeuralNetwork(LAYERS)
    NEURALNETWORK.train(inputs, labels, learning_rate, iterations, split_size, downsample_size)

    x = inputs
    y = labels
    xt = validation_inputs
    yt = validation_labels

    t2 = time.time()
    print('Training time: %.1fs' % (t2 - t1))

    training_accuracy = NEURALNETWORK.accuracy(x, y)
    validation_accuracy = NEURALNETWORK.accuracy(xt, yt)

    print("Training accuracy:", training_accuracy / x.shape[0])
    print("Validation accuracy:", validation_accuracy / xt.shape[0])

    NEURALNETWORK.plot()


def plot_mnist_digits(images):
    images_len = len(images)
    plot_sizes = [0, 0]
    while plot_sizes[0]*plot_sizes[1] < images_len:
        plot_sizes[0] += 1
        if plot_sizes[0]*plot_sizes[1] < images_len:
            plot_sizes[1] += 1
    fig = plt.figure()

    for i in range(images_len):
        ax = fig.add_subplot(plot_sizes[0], plot_sizes[1], i+1)
        ax.axis('off')
        ax.matshow(images[i], cmap=plt.get_cmap("Greys"))

    plt.show()


def load_random_digit(file):

    img = Image.open(file)
    img.load()
    data = np.array(img, dtype="int32")

    digit = data[:, :, 0]
    digit = np.reshape(digit, (1, 1, digit.shape[0], digit.shape[1])) / 255.0
    return digit




if __name__ == "__main__":

    learning_rate = 1e-1
    iterations = 3
    split_size = 50
    downsample_size = 3
    run(learning_rate, iterations, split_size, downsample_size)










