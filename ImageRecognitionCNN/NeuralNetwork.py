"""
Neural network for CNN
"""
import OneHot as onehot
import numpy as np
import matplotlib.pyplot as plt
import random


class NeuralNetwork:

    def __init__(self, layers):

        self.layers = layers

    def train(self, inputs, labels, learning_rate, iterations, split_size):
        N = labels.shape[0]
        batches = N // split_size
        one_hot_labels = onehot.one_hot(labels)

        for iter in range(iterations):

            for mini_batch in range(batches):

                input_vector_batch = inputs[mini_batch * split_size: split_size * (mini_batch + 1)]
                labels_batch = one_hot_labels[mini_batch * split_size: split_size * (mini_batch + 1)]

                tuple = []
                for i in range(input_vector_batch.shape[0]):
                    tuple.append((input_vector_batch[i], labels_batch[i]))

                random.shuffle(tuple)
                for i in range(len(tuple)):
                    input_vector_batch[i] = tuple[i][0]
                    labels_batch[i] = tuple[i][1]
                input_vector = input_vector_batch

                # count = -1
                for layer in self.layers:
                    input_vector = layer.forwards_pass(input_vector)
                    # count += 1
                    # print(count)
                    # print(input_vector.shape)

                output_vector_prediction = input_vector
                output_gradient = self.layers[-1].gradient(output_vector_prediction, labels_batch)

                # count = 9
                for layer in reversed(self.layers[:-1]):
                    output_gradient = layer.backwards_pass(output_gradient)
                    # count = count + 1
                    # print(count)
                    # print(output_gradient.shape)

                for layer in self.layers:
                    layer.update_parameters(learning_rate)

                net_loss = self.layers[-1].loss(output_vector_prediction, labels_batch)
                if mini_batch % 10 == 0:
                    print('iteration %i, loss %.9f, minibatch %i' % (iter, net_loss, mini_batch))

                learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * 0.01)))

    def plot_mnist_digit(self, image1):
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        ax.matshow(image1, cmap=plt.get_cmap('Greys'))
        ax = fig.add_subplot(2, 2, 2)
        ax.matshow(image1, cmap=plt.get_cmap('Greys'))
        ax = fig.add_subplot(2, 2, 3)
        ax.matshow(image1, cmap=plt.get_cmap('Greys'))
        ax = fig.add_subplot(2, 2, 4)
        ax.matshow(image1, cmap=plt.get_cmap('Greys'))
        plt.show()

    def predict(self, input_vector):

        for layer in self.layers:
            input_vector = layer.forwards_pass(input_vector)

        return input_vector

    def accuracy(self, input_vector, labels):

        prediction = self.predict(input_vector)

        acc = np.array([(prediction == labels)])
        acc_sum = np.sum(acc)
        N = input_vector.shape[0]
        avg_acc = acc_sum / N

        return avg_acc


