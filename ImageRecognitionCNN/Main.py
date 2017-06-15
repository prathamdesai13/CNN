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


def run(learning_rate, iterations, split_size):

    inputs, labels, validation_inputs, validation_labels, num_classes = initialize()

    validation_labels = onehot.one_hot(validation_labels)

    convLayer1 = ConvolutionLayer(8, 5, 1)
    reluLayer = ReLU(1)
    poolLayer1 = PoolingLayer((2, 2), 2)
    convLayer2 = ConvolutionLayer(16, 5, 1)
    reluLayer2 = ReLU(1)
    poolLayer2 = PoolingLayer((2, 2), 2)
    vectorizeLayer = Vectorize()
    hiddenLayer = HiddenLayer(10, 256)
    finalLayer = FinalLayer()

    LAYERS = [convLayer1, reluLayer, poolLayer1, convLayer2, reluLayer2, poolLayer2, vectorizeLayer, hiddenLayer,
              finalLayer]
    t1 = time.time()

    NEURALNETWORK = NeuralNetwork(LAYERS)
    NEURALNETWORK.train(inputs, labels, learning_rate, iterations, split_size)
    
    kernels_firstlayer = convLayer1.kernels
    kernel = convLayer2.kernels
    weights = hiddenLayer.weights
    bias = hiddenLayer.bias
    print("Second kernel shape:", kernel.shape)

    t2 = time.time()

    accuracy = NEURALNETWORK.accuracy(validation_inputs, validation_labels)

    t3 = time.time()

    print("Number of correct predictions:")
    print(accuracy)
    print('Time to train: %.1fs' % (t2 - t1))
    print('Time to test accuracy: %.1fs' % (t3 - t2))



if __name__ == "__main__":

    learning_rate = 0.05
    iterations = 1
    split_size = 50

    run(learning_rate, iterations, split_size)










