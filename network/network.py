"""
strangecamelcaselogin (me) "fork" of source code to this book.
    http://neuralnetworksanddeeplearning.com/

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

"""

from time import time, localtime
import _pickle

import numpy as np

import theano
import theano.tensor as T
from theano import pp

from .functions_ import size, safe_float2int

# Activation functions for neurons
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


def linear(z): return z


def ReLU(z): return T.maximum(0.0, z)


class Network:
    def __init__(self, layers, mini_batch_size, name=None):
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

        self.hyper_params = dict()
        self.name = name

    def SGD(self, training_data, validation_data, test_data, epochs, eta, lmbda=0.0):
        """
        Train the network using mini-batch stochastic gradient descent.

        :param eta: speed of training
        :param lmbda: parameter for L2 regularisation
        :return:
        """

        train_timer = time()
        self.save_hyper_params(epochs, eta, lmbda)

        num_training_batches = safe_float2int(size(training_data) / self.mini_batch_size)
        num_validation_batches = safe_float2int(size(validation_data) / self.mini_batch_size)
        num_test_batches = safe_float2int(size(test_data) / self.mini_batch_size)

        sharp_update = int(num_training_batches / 10)
        print("Start SGD training.")
        print("{0} epochs, {1} training batches per epoch.".format(epochs, num_training_batches))
        print("# - {} batches.\n".format(sharp_update))

        # data
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # ETA as function(epochs)
        f_eta = theano.shared(eta[0])
        dec_eta = (eta[0] - eta[1]) / epochs

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + 0.5 * lmbda * l2_norm_squared / num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param - T.cast(f_eta, 'float32') * grad) for param, grad in zip(self.params, grads)]

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

        vld_acc_history = []
        best_validation_accuracy = 0.0
        best_iteration = 0
        test_accuracy = 0
        for epoch in range(epochs):
            epoch_timer = time()
            print("Epoch {0: <2} ETA={1: <4}: ".format(epoch + 1, np.round(f_eta.get_value(), decimals=2)), end='',
                  flush=True)

            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if minibatch_index % sharp_update == 0:
                    print("#", end='', flush=True)

                train_mb(minibatch_index)  # train function call

                if (iteration + 1) % num_training_batches == 0:
                    validation_accuracy = np.mean([validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print(" : by {0:.1f} sec. | vld acc = {1:.2%}"
                          .format(time() - epoch_timer, validation_accuracy), end=' ')

                    vld_acc_history.append(validation_accuracy)

                    if validation_accuracy >= best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        test_accuracy = np.mean([test_mb_accuracy(j) for j in range(num_test_batches)])
                        print("(best) | cor acc = {0:.2%}".format(test_accuracy))
                    else:
                        print()
            # decrement ETA
            f_eta.set_value(f_eta.get_value() - dec_eta)

        print()
        print("Finished training network by {0:.1f} sec.".format(time() - train_timer))
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}."
              .format(best_validation_accuracy, best_iteration + 1))
        print("Corresponding test accuracy of {0:.2%}.".format(test_accuracy))

        return vld_acc_history

    def predict(self, data, index):
        """
        :param data: array of images
        :param index: index in this array
        :return: class of image
        """
        i = T.lscalar()
        _predict_batch = theano.function([i], self.layers[-1].y_out, givens={
            self.x: data[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]})

        batch_index = index // self.mini_batch_size
        num_in_batch = index % self.mini_batch_size

        return _predict_batch(batch_index)[num_in_batch]

    def save(self, path=None):
        """
        :param path: path to file, that will contain Network instance.
        :return:
        """
        if path is None:
            t = localtime()
            path = './{}-{}-{}_{}-{}-{}.net'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

        with open(path, 'wb') as f:
            _pickle.dump(self.__dict__, file=f)

        print('saved as {}'.format(path))

    @classmethod
    def load(cls, path):
        """
        example:
            net = Network.load('./2017-2-6_11-28-50.net')

        :param path: path to .net file
        :return: Network instance
        """
        with open(path, 'rb') as f:
            _dict = _pickle.load(file=f, encoding='latin1')

        obj = cls.__new__(cls)
        obj.__dict__.update(_dict)
        return obj

    def save_hyper_params(self, epochs, eta, lmbda):
        """
        Save hyper parameters in self.hyper_params.
        """
        self.hyper_params = {'epochs': epochs,
                             'mini_batch_size': self.mini_batch_size,
                             'eta': eta,
                             'lmbda': lmbda,
                             'name': self.name}

    def show_info(self):
        print('\ninfo:')
        for k in self.hyper_params.keys():
            print('{} = {}'.format(k, self.hyper_params[k]))
        print()
