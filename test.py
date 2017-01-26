#import theano
#from theano.tensor import shape
#import numpy as np

from functions_ import load_data_shared

from network import Network
from convolution import ConvPoolLayer
from fullyconnected import FullyConnectedLayer
from softmax import SoftmaxLayer


if __name__ == '__main__':
    """
    GPU = True
    if GPU:
        print("Trying to run under a GPU.")
        try:
            theano.config.device = 'gpu'
        except Exception as e:
            print(e)
        theano.config.floatX = 'float32'
    else:
        print("Running with a CPU")
    """

    training_data, validation_data, test_data = load_data_shared()
    print("Data loaded")

    '''
    mini_batch_size = 10
    net = Network([
        FullyConnectedLayer(n_in=784, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    net.SGD(training_data, 60, mini_batch_size, 0.1,
            validation_data, test_data)
    '''
    mini_batch_size = 5
    epochs = 3
    ETA = 0.1

    net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                          filter_shape=(20, 1, 5, 5),
                          poolsize=(2, 2)),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                          filter_shape=(40, 20, 5, 5),
                          poolsize=(2, 2)),
            FullyConnectedLayer(n_in=40*4*4, n_out=80),
            SoftmaxLayer(n_in=80, n_out=10)], mini_batch_size)

    net.SGD(training_data, epochs, mini_batch_size, ETA, validation_data, test_data)
    #net.test_mb_predictions()