from keras.datasets import mnist
from keras.utils import to_categorical, plot_model
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, GlobalMaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import keras
import keras.activations as activations
import keras.layers as layers

# from cnn_configs import Slim, Full

# %load_ext tensorboard
import tensorflow as tf

from kerastuner import HyperModel, HyperParameters
from kerastuner.tuners import Hyperband


import datetime

import numpy as np
import matplotlib.pyplot as plt


class SimpNet(HyperModel):
    def __init__(self, config) -> None:
        self.config = config

    def build(self, hp):
        self.net = Sequential()
        self.inference(hp)
        self.optimizer_net_compile(hp)
        return self.net

    def inference(self, hp):
        wi = hp.Choice('weight_init', values=['HeUniform', 'HeNormal', 'GlorotUniform', 'GlorotNormal'])
        c = self.config
        if c.dropout:
            dr = hp.Float('dropout_rate', 0.65, 0.95)
        else:
            dr = hp.Fixed('dropout_rate', 0.8)

        f0 = hp.Choice('filters0', values=[24, 28, 30, 34, 36], default=30)
        f1 = hp.Choice('filters1', values=[36, 38, 40, 42, 44], default=40)
        f2 = hp.Choice('filters2', values=[46, 48, 50, 52, 54], default=50)
        f3 = hp.Choice('filters3', values=[54, 56, 58, 60, 62], default=58)
        f4 = hp.Choice('filters4', values=[66, 68, 70, 72, 74], default=70)
        f5 = hp.Choice('filters5', values=[86, 88, 90, 82, 94], default=90)

        self.conv_relu_bn_dropout(filters=f0, input_shape=c.INPUT_SHAPE, dropout=c.dropout, weight_init=wi)
        self.conv_relu_bn_dropout(filters=f1, dropout_rate=dr, dropout=c.dropout, weight_init=wi)
        self.conv_relu_bn_dropout(filters=f1, dropout_rate=dr, dropout=c.dropout, weight_init=wi)
        self.conv_relu_bn_dropout(filters=f1, dropout_rate=dr, dropout=c.dropout, weight_init=wi)
        self.conv_relu_bn_dropout(filters=f2, dropout_rate=dr, dropout=c.dropout, weight_init=wi)
        self.saf_pool()
        self.conv_relu_bn_dropout(filters=f2, dropout_rate=dr, dropout=c.dropout, weight_init=wi)
        self.conv_relu_bn_dropout(filters=f2, dropout_rate=dr, dropout=c.dropout, weight_init=wi)
        self.conv_relu_bn_dropout(filters=f2, dropout_rate=dr, dropout=c.dropout, weight_init=wi)
        self.conv_relu_bn_dropout(filters=f2, dropout_rate=dr, dropout=c.dropout, weight_init=wi)
        self.conv_relu_bn_dropout(filters=f3, dropout_rate=dr, dropout=c.dropout, weight_init=wi)
        self.saf_pool()
        self.conv_relu_bn_dropout(filters=f3, dropout_rate=dr, dropout=c.dropout, weight_init=wi)
        self.conv_relu_bn_dropout(filters=f4, dropout_rate=dr, dropout=c.dropout, weight_init=wi)
        self.conv_relu_bn_dropout(filters=f5, dropout_rate=dr, dropout=c.dropout, weight_init=wi)
        self.net.add(GlobalMaxPooling2D())
        if c.dropout:
            self.net.add(Dropout(rate=dr))
        self.net.add(Flatten())
        self.net.add(Dense(c.OUTPUT_SIZE, activation='softmax'))

    def optimizer_net_compile(self, hp):
        c = self.config

        base_lr = hp.Float('base_lr', 0.1, 0.5, sampling='log', default=0.2)
        decay_steps = hp.Int('decay_steps', 2500, 10000, sampling='linear', default=7185)
        decay_rate = hp.Float('decay_rate', 0.1, 0.95, sampling='linear', default=0.1)
        lr_momentum = hp.Float('lr_momentum', 0.9, 0.99, sampling='log', default=0.95)

        # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        #     base_lr,
        #     decay_steps,
        #     decay_rate,
        #     staircase=True,
        # )

        # default 0.30% misclassified
        boundaries = [5000, 9500, 22000, 29600, 32000, 37000]
        values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

        lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries,
            values,
        )
        opt = keras.optimizers.Adadelta(learning_rate=lr_schedule, rho=lr_momentum, epsilon=c.EPSILON)
        # opt = keras.optimizers.Adadelta()

        self.net.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')

    @staticmethod
    def get_callbacks():
        checkpoint_filepath = 'checkpoint/'
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            save_best_only=True,
        )

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=6)

        callbacks = [model_checkpoint_callback, tensorboard_callback, early_stopping]

        return callbacks

    def conv_relu_bn_dropout(
        self,
        filters,
        kernel_size=(3, 3),
        padding='same',
        bn_momentum=0.99,
        dropout_rate=0.2,
        dropout=False,
        input_shape=None,
        bn_before_relu=True,
        weight_init='HeUniform',
    ):
        if input_shape is None:
            self.net.add(
                Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding=padding,
                    kernel_initializer=weight_init,
                )
            )
        else:
            self.net.add(
                Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding=padding,
                    kernel_initializer=weight_init,
                    input_shape=input_shape,
                )
            )
        if bn_before_relu:
            # self.net.add(layers.Activation(activations.relu))
            self.net.add(BatchNormalization(momentum=bn_momentum))
            # self.net.add(layers.PReLU())
            self.net.add(layers.ReLU())
        else:
            self.net.add(layers.ReLU())
            # self.net.add(layers.PReLU())
            self.net.add(BatchNormalization(momentum=bn_momentum))

        if dropout:
            self.net.add(Dropout(rate=dropout_rate))

    def saf_pool(self, pool_size=(2, 2), strides=2, padding='valid', dropout_rate=0.2):
        self.net.add(MaxPool2D(pool_size=pool_size, strides=strides, padding=padding))
        self.net.add(Dropout(rate=dropout_rate))


def error_rate(model_path, x_test, labels_test):
    net = load_model(model_path)
    outputs = net.predict(x_test)
    labels_predicted = np.argmax(outputs, axis=1)
    misclassified = sum(labels_predicted != labels_test)
    print('pct misclassified = ', 100 * misclassified / labels_test.size)


def preprocess():
    (x_train, labels_train), (x_test, labels_test) = mnist.load_data()

    # convert to float type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # one hot encoding
    y_train = to_categorical(labels_train, 10)
    y_test = to_categorical(labels_test, 10)

    n_pixels = x_train.shape[1]

    # plt.figure()
    # plt.imshow(x_train[0])
    # plt.show()

    x_train = x_train.reshape(x_train.shape[0], n_pixels, n_pixels, 1)
    x_test = x_test.reshape(x_test.shape[0], n_pixels, n_pixels, 1)

    return x_train, x_test, labels_train, y_train, y_test, labels_test


if __name__ == "__main__":
    x_train, x_test, labels_train, y_train, y_test, labels_test = preprocess()

    simpnet = SimpNet(config=Slim())
    # simpnet = SimpNet(config=Full())

    mode = 'test'

    if mode == 'tune':
        tuner = Hyperband(
            simpnet,
            objective='val_accuracy',
            max_epochs=15,
            hyperband_iterations=3,
        )

        tuner.search(
            x_train,
            y_train,
            epochs=20,
            batch_size=100,
            validation_data=(x_test, y_test),
            callbacks=SimpNet.get_callbacks(),
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = tuner.hypermodel.build(best_hps)

        history = model.fit(
            x_train,
            y_train,
            epochs=50,
            validation_data=(x_test, y_test),
            callbacks=SimpNet.get_callbacks(),
            verbose=2,
        )

    elif mode == 'test':
        hp = HyperParameters()
        # best 1
        # hp.Fixed('weight_init', value='GlorotUniform')
        # hp.Fixed('base_lr', value=0.21145)
        # hp.Fixed('decay_steps', value=7185)
        # hp.Fixed('decay_rate', value=0.115)
        # hp.Fixed('lr_momentum', value=0.91074)

        # # best2
        # hp.Fixed('weight_init', value='HeUniform')
        # hp.Fixed('base_lr', value=0.27872)
        # hp.Fixed('decay_steps', value=3805)
        # hp.Fixed('decay_rate', value=0.44912)
        # hp.Fixed('lr_momentum', value=0.93493)

        # manual tune
        hp.Fixed('weight_init', value='GlorotUniform')
        hp.Fixed('base_lr', value=0.1)
        # hp.Fixed('decay_steps', value=2500)
        # hp.Fixed('decay_rate', value=0.1)
        hp.Fixed('lr_momentum', value=0.9)

        f0 = hp.Fixed('filters0', value=30)
        f1 = hp.Fixed('filters1', value=40)
        f2 = hp.Fixed('filters2', value=50)
        f3 = hp.Fixed('filters3', value=58)
        f4 = hp.Fixed('filters4', value=70)
        f5 = hp.Fixed('filters5', value=90)

        model = simpnet.build(hp)

        history = model.fit(
            x_train,
            y_train,
            epochs=50,
            batch_size=100,
            validation_data=(x_test, y_test),
            callbacks=SimpNet.get_callbacks(),
            verbose=2,
        )
    elif mode == 'final':
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))

        hp = HyperParameters()
        hp.Fixed('weight_init', value='GlorotUniform')
        hp.Fixed('base_lr', value=0.21145)
        hp.Fixed('decay_steps', value=7185)
        hp.Fixed('decay_rate', value=0.115)
        hp.Fixed('lr_momentum', value=0.91074)
        model = simpnet.build(hp)

        history = model.fit(
            x,
            y,
            epochs=50,
            batch_size=100,
            callbacks=SimpNet.get_callbacks(),
            verbose=2,
        )
    else:
        raise Exception('unrecognised training mode!')

    model.save('SimpNet_best.h5')

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
