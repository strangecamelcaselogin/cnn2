from random import randint
import matplotlib.pyplot as plt
import theano

from network import Network

from network.layers import ConvPoolLayer
from network.layers import FullyConnectedLayer
from network.layers import SoftmaxLayer

from network import ReLU, sigmoid, tanh

from network.functions_ import load_data_shared, size


def show_img(data, num):
    images, labels = data[0].get_value(), data[1].eval()

    im = images[num].reshape((28, 28))
    lb = labels[num]
    predict = net.predict(data[0], num)

    plt.imshow(im, cmap='Greys', interpolation='none')
    plt.title('predict: {}, label: {}'.format(predict, lb))
    plt.show()


def plot_vld_acc(history, epochs=40, low_percent=0.9):
    plt.axis([0, epochs, low_percent, 1])
    plt.plot(history)
    plt.title("Validation accuracy / Epoch.")
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
    ETA = (0.75, 0.2)
    lmbda = 0.1
    dropout = 0.5

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
                            p_dropout=dropout),
        SoftmaxLayer(n_in=100,
                     n_out=10,
                     p_dropout=dropout)], mini_batch_size)

    h = net.SGD(training_data, validation_data, test_data, epochs, ETA, lmbda=lmbda, sharp_update=100)
    net.save()

    # net = Network.load()

    net.show_info()

    plot_vld_acc(h)
    print(h)

    # for i in [randint(0, 10000) for _ in range(25)]:
    #    show_img(validation_data, i)

    # TODO early stop algorithm
    # TODO more reports
