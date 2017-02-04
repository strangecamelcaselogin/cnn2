# import theano
# from theano.tensor import shape
# import numpy as np

from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

from functions_ import load_data_shared
from functions_ import ReLU

from network import Network
from convolution import ConvPoolLayer
from fullyconnected import FullyConnectedLayer
from softmax import SoftmaxLayer


def show():
    pass
    """
        img = None
        for i in ttt:
            im = training_data[0].get_value()[i].reshape((28, 28))
            if img is None:
                img = plt.imshow(im, cmap='Greys', interpolation='none')
            else:
                img.set_data(im)
            plt.pause(2.)
            print(net.test_mb_predictions(i))
            plt.draw()

    """


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
    print("Data loaded.")

    '''
    mini_batch_size = 10
    net = Network([
        FullyConnectedLayer(n_in=784, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    net.SGD(training_data, 60, mini_batch_size, 0.1,
            validation_data, test_data)
    '''
    mini_batch_size = 10
    epochs = 8
    ETA = 0.1

    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40 * 4 * 4,
                            n_out=100,
                            activation_fn=ReLU,
                            p_dropout=0.25),
        SoftmaxLayer(n_in=100,
                     n_out=10,
                     p_dropout=0.25)], mini_batch_size)

    net.SGD(training_data, validation_data, epochs, ETA, test_data)

    # TODO test_data
    # print(net.test_mb_predictions(150))
    # [3 3 3 3 0 3 3 3 3 3]
    # [1 9 8 3 0 7 2 7 9 4]
    # im150 = training_data[0].get_value()[150].reshape((28, 28))
    # plt.imshow(im150, cmap='Greys', interpolation='none')
