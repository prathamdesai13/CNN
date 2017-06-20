"""
Neural network for CNN
"""
import OneHot as onehot
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, layers):

        self.layers = layers

    def train(self, inputs, labels, learning_rate, iterations, split_size):
        N = labels.shape[0]
        N = 2000
        batches = N // split_size
        loss = []
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
                loss.append(net_loss)
                if mini_batch % 10 == 0:
                    print('iteration %i, loss %.9f, minibatch %i' % (iter, net_loss, mini_batch))
            #learning_rate = learning_rate * (learning_rate / (learning_rate + (learning_rate * 0.01)))

        plt.plot(loss)
        plt.show()

    def predict(self, input_vector):
        print("Starting predictions")
        count = 0
        for layer in self.layers:
            input_vector = layer.forwards_pass(input_vector)
            count = count + 1
        return input_vector

    def accuracy(self, input_vector, labels):

        prediction = self.predict(input_vector)
        count = 0
        for i in range(prediction.shape[0]):
            print(i)
            print(prediction[i])
            print(labels[i])
            if np.array_equal(prediction[i], labels[i]):
                count = count + 1
            print("\n")

        return count

    def another_accuracy(self, input_vector, labels):
        prediction = self.predict(input_vector)

        print(prediction[5])
        print(labels[5])
        pred_indices = [np.argmax(prediction, axis=1)]
        label_indices = [np.argmax(labels, axis=1)]
        count = 0
        for i in range(len(pred_indices[0])):
            if pred_indices[0][i] == label_indices[0][i]:
                count = count + 1
        return count

