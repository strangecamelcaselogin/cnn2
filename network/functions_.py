import _pickle
import gzip

import numpy as np
import theano
import theano.tensor as T


def size(data):
    """Return the size of the dataset `data`."""
    return data[0].get_value(borrow=True).shape[0]


def safe_float2int(x):
    if not x.is_integer():
        raise ValueError("Result of division not integer (as expected).")
    else:
        return int(x)


def load_data_shared(filename="./MNIST/mnist.pkl.gz"):
    def _shared(data):
        shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = _pickle.load(file=f, encoding='latin1')

    return [_shared(training_data), _shared(validation_data), _shared(test_data)]
