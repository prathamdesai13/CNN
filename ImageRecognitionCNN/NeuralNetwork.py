"""
Neural network for CNN
"""
import OneHot as onehot
import numpy as np


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
                if mini_batch % 1 == 0:
                    print('iteration %i, loss %.9f, minibatch %i' % (iter, net_loss, mini_batch))

                learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * 0.01)))

    def predict(self, input_vector):

        for layer in self.layers:
            input_vector = layer.forwards_pass(input_vector)

        return input_vector

    def accuracy(self, input_vector, labels):

        prediction = self.predict(input_vector)
        count = 0
        for i in range(prediction.shape[0]):
            sub_count = 0
            for j in range(prediction.shape[1]):
                if prediction[i, j] == labels[i, j]:
                    sub_count += 1
            count += sub_count // 10

        return count


