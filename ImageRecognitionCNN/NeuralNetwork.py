"""
Neural network for CNN
"""

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:

    def __init__(self, layers):

        self.layers = layers
        self.losses = []

    def train(self, inputs, labels, learning_rate, iterations, split_size, downsample_size):

        N = labels.shape[0] // downsample_size
        bool = True
        print(N)
        batches = N // split_size
        count = 0

        for iter in range(iterations):

            for mini_batch in range(batches):

                input_vector_batch = inputs[mini_batch * split_size: split_size * (mini_batch + 1)]
                labels_batch = labels[mini_batch * split_size: split_size * (mini_batch + 1)]

                input_vector = input_vector_batch

                for layer in self.layers:
                    if count == 0:
                        bool = True
                    else:
                        bool = False

                    input_vector = layer.forwards_pass(input_vector, bool)
                    # count += 1
                    # print(count)
                    # print(input_vector.shape)

                output_vector_prediction = input_vector
                self.layers[-1].gradient(output_vector_prediction, labels_batch)

                # count = 9
                output_gradient = 1
                for layer in reversed(self.layers):
                    output_gradient = layer.backwards_pass(output_gradient)
                    # count = count + 1
                    # print(count)
                    # print(output_gradient.shape)

                for layer in self.layers:
                    layer.update_parameters(learning_rate)

                net_loss = self.layers[-1].loss(output_vector_prediction, labels_batch)
                self.losses.append(net_loss)
                count = 1
                if mini_batch % 1 == 0:
                    print('iteration %i, loss %.9f, minibatch %i' % (iter, net_loss, mini_batch))
                #learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * 0.01)))
            #learning_rate *= 0.99

    def predict(self, input_vector):
        print("Predictions:")

        for layer in self.layers:
            input_vector = layer.forwards_pass(input_vector, bool=False)

        return input_vector

    def accuracy(self, input_vector, labels):
        prediction = self.predict(input_vector)
        pred_indices = [np.argmax(prediction, axis=1)]
        label_indices = [np.argmax(labels, axis=1)]
        count = 0
        #print(pred_indices[0:1])
        #print(label_indices[0:1])
        for i in range(len(pred_indices[0])):
            if pred_indices[0][i] == label_indices[0][i]:
                count = count + 1
        return count

    def plot(self):

        plt.plot(self.losses)
        plt.show()
