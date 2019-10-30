from keras import models
from keras.datasets import mnist
from keras import layers
from keras.utils import to_categorical
from keras import optimizers
from datetime import datetime


def run(hidden_size=300, batch_size=128, rate=0.1, epochs=20):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    network = models.Sequential()
    network.add(layers.Dense(hidden_size, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
    optimizer = optimizers.SGD(rate)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    time_start = datetime.now()
    network.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
    time = datetime.now() - time_start
    train_result = network.evaluate(train_images, train_labels)
    test_result = network.evaluate(test_images, test_labels)
    return train_result, test_result, time


if __name__ == '__main__':
    train_result, test_result, time = run()
    print('Train loss: ', train_result[0])
    print('Train accuracy: ', train_result[1])
    print('Test loss: ', test_result[0])
    print('Test accuracy: ', test_result[1])
    print('Time: ', time)



