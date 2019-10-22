from keras.datasets import mnist
from keras.utils import to_categorical
from datetime import datetime
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def relu(x):
    return np.vectorize(relu_s)(x)


def relu_der(x):
    return np.vectorize(relu_der_s)(x)


def relu_der_s(x):
    return 1 if x > 0 else 0


def relu_s(x):
    return x if x > 0 else 0


def shuffle_samples(a, b):
    random_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(random_state)
    np.random.shuffle(b)
    return a, b


class Network:
    def __init__(self, hidden_nodes_num, output_nodes_num, rate, debug=False):
        self.__input_nodes_num = 0
        self.__hidden_nodes_num = hidden_nodes_num
        self.__output_nodes_num = output_nodes_num
        self.__nodes = {}
        self.__wh = np.array([])
        self.__ws = np.array([])
        self.__rate = rate
        self.__batch_size = 0
        self.debug = debug

    def fit(self, X, Y, batch_size, number_epochs):
        np.random.seed(1)
        self.__batch_size = batch_size
        self.__input_nodes_num = X.shape[1]
        self.__init_weights()

        for epoch in range(number_epochs):
            X, Y = shuffle_samples(X, Y)
            for i in range(0, X.shape[0], self.__batch_size):
                self.__forward_prop(X[i:i + self.__batch_size])
                self.__back_prop(X[i:i + self.__batch_size], Y[i:i + self.__batch_size])

    def __init_weights(self):
        coef_h = 2.0 / np.sqrt(self.__input_nodes_num + self.__hidden_nodes_num)
        coef_s = 2.0 / np.sqrt(self.__hidden_nodes_num + self.__output_nodes_num)
        self.wh = coef_h * np.random.randn(self.__hidden_nodes_num, self.__input_nodes_num)
        self.ws = coef_s * np.random.randn(self.__output_nodes_num, self.__hidden_nodes_num)

    def __forward_prop(self, input):
        X1p = np.dot(self.wh, input.T)
        X1 = relu(X1p)
        self.__nodes['X1p'] = X1p
        self.__nodes['X1'] = X1

        X2p = np.dot(self.ws, X1)
        X2 = softmax(X2p)
        self.__nodes['X2p'] = X2p
        self.__nodes['X2'] = X2

    def __back_prop(self, X, Y):
        d2 = Y.T - self.__nodes['X2']
        dws = np.dot(d2, self.__nodes['X1'].T) / self.__batch_size

        d1 = np.dot(self.ws.T, d2) * relu_der(self.__nodes['X1p'])
        dwh = np.dot(d1, X) / self.__batch_size

        self.ws = self.ws + self.__rate * dws
        self.wh = self.wh + self.__rate * dwh

    def evaluate(self, X, Y):
        self.__forward_prop(X)
        crossentropy = -np.sum(Y * np.log(self.__nodes['X2'].T)) / X.shape[0]

        result_net = np.argmax(self.__nodes['X2'], axis=0)
        result_real = np.argmax(Y, axis=1)
        accuracy = (result_net == result_real).mean()

        return crossentropy, accuracy


def run(hidden_size=240, batch_size=128, rate=0.3, epochs=5):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    net = Network(hidden_size, 10, rate)

    time_start = datetime.now()
    net.fit(train_images, train_labels, batch_size=batch_size, number_epochs=epochs)
    time = datetime.now() - time_start
    train_result = net.evaluate(train_images, train_labels)
    test_result = net.evaluate(test_images, test_labels)
    return train_result, test_result, time


if __name__ == '__main__':
    train_result, test_result, time = run()
    print('Train loss: ', train_result[0])
    print('Train accuracy: ', train_result[1])
    print('Test loss: ', test_result[0])
    print('Test accuracy: ', test_result[1])
    print('Time: ', time)

















