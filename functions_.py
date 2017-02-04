#import _pickle
#import cPickle
import _pickle
import gzip

import numpy as np
import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams


# Activation functions for neurons
def linear(z):
    return z


def ReLU(z):
    return T.maximum(0.0, z)


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    return layer * T.cast(mask, theano.config.floatX)


def size(data):
    """Return the size of the dataset `data`."""
    return data[0].get_value(borrow=True).shape[0]


def load_data_shared(filename="./MNIST/mnist.pkl.gz"):
    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = _pickle.load(file=f, encoding='latin1')
    #f = gzip.open(filename, 'rb')
    #training_data, validation_data, test_data = cPickle.load(f)
    #f.close()

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    return [shared(training_data), shared(validation_data), shared(test_data)]
