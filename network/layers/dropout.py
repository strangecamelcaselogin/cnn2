import numpy as np

import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams


class __Dropout:
    def __call__(self, layer, p_dropout):
        srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
        mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
        return layer * T.cast(mask, theano.config.floatX)

DropoutLayer = __Dropout()
