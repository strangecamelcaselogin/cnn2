from random import randint
import theano
import matplotlib.pyplot as plt

from network import Network
from convolution import ConvPoolLayer
from fullyconnected import FullyConnectedLayer
from softmax import SoftmaxLayer

from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from functions_ import load_data_shared, size
from functions_ import ReLU


def check_show(size, data):
    """
    img = None
    for i in test_set:
        im = training_data[0].get_value()[i].reshape((28, 28))
        if img is None:
            img = plt.imshow(im, cmap='Greys', interpolation='none')
        else:
            img.set_data(im)
        plt.pause(2.)
        print(net.test_mb_predictions(i))
        plt.draw()
    """
    pass


def show_img(img_data, num):
    print(net.predict(test_data[0], num))  # TODO label or title
    im = img_data[num].reshape((28, 28))
    plt.imshow(im, cmap='Greys', interpolation='none')
    plt.show()

if __name__ == '__main__':

    '''
    # Простой перцептрон
    net = Network([
        FullyConnectedLayer(n_in=784, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

    net.SGD(training_data, 60, mini_batch_size, 0.1,
            validation_data, test_data)
    '''

    training_data, validation_data, test_data = load_data_shared()
    print("Data loaded, sizes: train={0}, valid={1}, test={2}.\n"
          .format(size(training_data), size(validation_data), size(test_data)))

    mini_batch_size = 50
    epochs = 40
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
    net.SGD(training_data, validation_data, test_data, epochs, ETA)

    #  net.show_info()
    net.save()

    # net = Network.load('./2017-2-9_10-12-15.net')

    # test_num = 9
    # show_img(test_data[0].get_value(), test_num)

    for i in [randint(0, 10000) for _ in range(25)]:
        show_img(test_data[0].get_value(), i)

    # TODO give name to net
    # TODO early stop
    # TODO ETA as function of epoch
