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

    x = inputs
    y = labels
    xt = validation_inputs
    yt = validation_labels

    conv1 = ConvolutionLayer(32, 5, 1)
    pool1 = PoolingLayer([2, 2], 2)
    conv2 = ConvolutionLayer(16, 5, 1)
    pool2 = PoolingLayer([2, 2], 2)
    relu1 = ReLU()
    relu2 = ReLU()
    relu3 = ReLU()


    vectorizeLayer = Vectorize()
    hiddenLayer = HiddenLayer(100, 4608)
    hiddenLayer2 = HiddenLayer(10, 100)
    finalLayer = FinalLayer()

    LAYERS = [conv1, pool1, relu1,
              vectorizeLayer,
              hiddenLayer, relu3,
              hiddenLayer2,
              finalLayer]

    t1 = time.time()

    NEURALNETWORK = NeuralNetwork(LAYERS)

    NEURALNETWORK.train(inputs, labels, learning_rate, iterations, split_size, downsample_size)

    weight1 = hiddenLayer.weights
    #bias1 = hiddenLayer.bias
    #weight2 = hiddenLayer2.weights
    #bias2 = hiddenLayer2.bias

    #np.savetxt('WEIGHTS1', weight1)
    #np.savetxt('BIAS1', bias1)
    #np.savetxt('WEIGHTS2', weight2)
    #np.savetxt('BIAS2', bias2)



    t2 = time.time()
    print('Training time: %.1fs' % (t2 - t1))

    #training_accuracy = NEURALNETWORK.accuracy(x, y)
    #validation_accuracy = NEURALNETWORK.accuracy(xt, yt)

    #print("Training accuracy:", training_accuracy / x.shape[0])
    #print("Validation accuracy:", validation_accuracy / xt.shape[0])

    NEURALNETWORK.plot()
    digit = xt[8]
    digit_label = yt[8]
    pred = NEURALNETWORK.predict(digit.reshape((1, 1, 28, 28)))
    print(pred)
    print(digit_label)

    list = [weight1[i].reshape((28, 28)) for i in range(10)]
    plot_mnist_digits(list)






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
        ax.matshow(images[i], cmap=plt.get_cmap('Greys'))

    plt.show()


def load_random_digit(file):

    img = Image.open(file)
    img.load()
    data = np.array(img, dtype="int32")
    data = data[:, :, 0]
    digit = np.reshape(data, (1, 1, data.shape[0], data.shape[1])) / 255.0
    return digit




if __name__ == "__main__":

    learning_rate = 3e-3
    iterations = 1
    split_size = 32
    downsample_size = 1
    run(learning_rate, iterations, split_size, downsample_size)










