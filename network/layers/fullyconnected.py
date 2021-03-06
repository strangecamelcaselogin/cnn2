import numpy as np

import theano
import theano.tensor as T

from ..network import sigmoid
from .dropout import DropoutLayer


class FullyConnectedLayer:
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout

        # Initialize weights and biases
        self.w = theano.shared(np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                                          dtype=theano.config.floatX), name='w', borrow=True)

        self.b = theano.shared(np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                                          dtype=theano.config.floatX), name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_input(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn((1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = DropoutLayer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        """Return the accuracy for the mini-batch."""
        return T.mean(T.eq(y, self.y_out))
