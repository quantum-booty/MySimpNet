from keras.datasets import mnist
from keras.utils import to_categorical, plot_model
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, GlobalMaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import keras
import keras.activations as activations
import keras.layers as layers
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


def preprocess():
    (x_train, labels_train), (x_test, labels_test) = mnist.load_data()

    # one hot encoding
    y_train = to_categorical(labels_train, 10)
    y_test = to_categorical(labels_test, 10)

    # convert to float type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    n_pixels = x_train.shape[1]

    # plt.figure()
    # plt.imshow(x_train[0])
    # plt.show()

    x_train = x_train.reshape(x_train.shape[0], n_pixels, n_pixels, 1)
    x_test = x_test.reshape(x_test.shape[0], n_pixels, n_pixels, 1)

    return x_train, x_test, labels_train, y_train, y_test, labels_test


def plot(x_train):
    # create a grid of 3x3 images
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_train[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == "__main__":
    x_train, x_test, labels_train, y_train, y_test, labels_test = preprocess()

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=30,
        preprocessing_function=None,
    )
    datagen.fit(x_train)

    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
        plot(x_batch)
        break
