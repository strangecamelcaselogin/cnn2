"""
strangecamelcaselogin (me) "fork" of source code to this book.
    http://neuralnetworksanddeeplearning.com/

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

"""

from time import time

import numpy as np
import theano
import theano.tensor as T
from theano import pp

from functions_ import size

from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


# Main class used to construct and train networks
class Network:
    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_input(self.x, self.x, self.mini_batch_size)

        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_input(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)

        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, validation_data, test_data, epochs, eta, lmbda=0.0):
        """
        Train the network using mini-batch stochastic gradient descent.

        :param eta: speed of training
        :param lmbda: parameter for L2 regularisation
        :return:
        """

        train_timer = time()

        num_training_batches = int(size(training_data) / self.mini_batch_size)
        num_validation_batches = int(size(validation_data) / self.mini_batch_size)
        num_test_batches = int(size(test_data) / self.mini_batch_size)

        print("Start SGD training.")
        print("{0} epochs, {1} training batches per epoch.".format(epochs, num_training_batches))
        print("# - 1000 batches.\n")

        # data
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + 0.5 * lmbda * l2_norm_squared / num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param - eta * grad) for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar()  # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x: training_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y: training_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: validation_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y: validation_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y: test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })

        best_validation_accuracy = 0.0
        best_iteration = 0
        test_accuracy = 0
        for epoch in range(epochs):
            epoch_timer = time()
            print("Epoch {0}: ".format(epoch), end='', flush=True)

            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if minibatch_index % 1000 == 0:
                    print("#", end='', flush=True)

                train_mb(minibatch_index)

                if (iteration + 1) % num_training_batches == 0:
                    validation_accuracy = np.mean([validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print(" : by {0:.1f} sec. | vld acc = {1:.2%}"
                          .format(time() - epoch_timer, validation_accuracy), end=' ')

                    if validation_accuracy >= best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        test_accuracy = np.mean([test_mb_accuracy(j) for j in range(num_test_batches)])
                        print("(best) | cor acc = {0:.2%}".format(test_accuracy))
                    else:
                        print()

        print()
        print("Finished training network by {0:.1f} sec.".format(time() - train_timer))
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}."
              .format(best_validation_accuracy, best_iteration + 1))
        print("Corresponding test accuracy of {0:.2%}.".format(test_accuracy))

    def predict(self, data, index):
        """
        :param data: array of images
        :param index: index in data
        :return: class of image
        """
        i = T.lscalar()
        _predict_batch = theano.function([i], self.layers[-1].y_out, givens={
            self.x: data[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]})

        batch_index = index // self.mini_batch_size
        num_in_batch = index % self.mini_batch_size

        return _predict_batch(batch_index)[num_in_batch]

    def save(self, path):
        pass

    def load(self, path):
        pass
