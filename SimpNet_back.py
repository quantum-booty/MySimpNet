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
from ray import tune
import ray


import datetime

import numpy as np
import matplotlib.pyplot as plt


class TuneReporterCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.iteration = 0
        super(TuneReporterCallback, self).__init__()

    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        tune.report(keras_info=logs, mean_accuracy=logs.get("accuracy"), mean_loss=logs.get("loss"))


class SimpNet:
    def __init__(self, hyper_space, config) -> None:
        self.config = config
        self.net = Sequential()

        self.hyper_space = hyper_space

    def build_network_graph(self):
        self.inference()
        self.optimizer_net_compile()

    def inference1(self):
        c = self.config
        self.conv_relu_bn_dropout(
            filters=c.CONV1_NUM_FILTERS, input_shape=c.INPUT_SHAPE, dropout=False, bn_before_relu=False
        )
        self.conv_relu_bn_dropout(filters=c.CONV2_NUM_FILTERS, dropout=False, bn_before_relu=False)
        self.conv_relu_bn_dropout(filters=c.CONV3_NUM_FILTERS, dropout=False, bn_before_relu=False)
        self.conv_relu_bn_dropout(filters=c.CONV4_NUM_FILTERS, dropout=False, bn_before_relu=False)
        self.conv_relu_bn_dropout(filters=c.CONV5_NUM_FILTERS, dropout=False, bn_before_relu=False)
        self.saf_pool()
        self.conv_relu_bn_dropout(filters=c.CONV6_NUM_FILTERS, dropout=False, bn_before_relu=False)
        self.conv_relu_bn_dropout(filters=c.CONV7_NUM_FILTERS, dropout=False, bn_before_relu=False)
        self.conv_relu_bn_dropout(filters=c.CONV8_NUM_FILTERS, dropout=False, bn_before_relu=False)
        self.conv_relu_bn_dropout(filters=c.CONV9_NUM_FILTERS, dropout=False, bn_before_relu=False)
        self.conv_relu_bn_dropout(filters=c.CONV10_NUM_FILTERS, dropout=False, bn_before_relu=False)
        self.saf_pool()
        self.conv_relu_bn_dropout(filters=c.CONV11_NUM_FILTERS, dropout=False, bn_before_relu=False)
        self.conv_relu_bn_dropout(filters=c.CONV12_NUM_FILTERS, dropout=False, bn_before_relu=False)
        self.conv_relu_bn_dropout(filters=c.CONV13_NUM_FILTERS, dropout=False, bn_before_relu=False)
        self.net.add(GlobalMaxPooling2D())
        # self.saf_pool(global_=True)
        self.net.add(Flatten())
        self.net.add(Dense(c.OUTPUT_SIZE, activation='softmax'))

    def inference(self):
        c = self.config
        self.conv_relu_bn_dropout(filters=c.CONV1_NUM_FILTERS, input_shape=c.INPUT_SHAPE, dropout=False)
        self.conv_relu_bn_dropout(filters=c.CONV2_NUM_FILTERS, dropout=False)
        self.conv_relu_bn_dropout(filters=c.CONV3_NUM_FILTERS, dropout=False)
        self.conv_relu_bn_dropout(filters=c.CONV4_NUM_FILTERS, dropout=False)
        self.conv_relu_bn_dropout(filters=c.CONV5_NUM_FILTERS, dropout=False)
        self.saf_pool()
        self.conv_relu_bn_dropout(filters=c.CONV6_NUM_FILTERS, dropout=False)
        self.conv_relu_bn_dropout(filters=c.CONV7_NUM_FILTERS, dropout=False)
        self.conv_relu_bn_dropout(filters=c.CONV8_NUM_FILTERS, dropout=False)
        self.conv_relu_bn_dropout(filters=c.CONV9_NUM_FILTERS, dropout=False)
        self.conv_relu_bn_dropout(filters=c.CONV10_NUM_FILTERS, dropout=False)
        self.saf_pool()
        self.conv_relu_bn_dropout(filters=c.CONV11_NUM_FILTERS, dropout=False)
        self.conv_relu_bn_dropout(filters=c.CONV12_NUM_FILTERS, dropout=False)
        self.conv_relu_bn_dropout(filters=c.CONV13_NUM_FILTERS, dropout=False)
        self.net.add(GlobalMaxPooling2D())
        # self.saf_pool(global_=True)
        self.net.add(Flatten())
        self.net.add(Dense(c.OUTPUT_SIZE, activation='softmax'))

    def optimizer_net_compile(self):
        c = self.config
        h = self.hyper_space

        # learning_rates = [h['base_lr'] * h['weight_decay'] ** i for i in range(0, len(c.BOUNDARIES) + 1)]
        # lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=c.BOUNDARIES, values=learning_rates)
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            h['base_lr'],
            decay_steps=h['decay_steps'],
            decay_rate=h['decay_rate'],
        )
        opt = keras.optimizers.Adadelta(learning_rate=lr_schedule, rho=h['lr_momentum'], epsilon=c.EPSILON)
        # opt = keras.optimizers.Adadelta()

        self.net.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')

    def fit(self, save_path, x_train, y_train, x_test, y_test, epochs, batch_size=100, ray_tune=False):
        checkpoint_filepath = 'checkpoint/'
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='accuracy',
            mode='max',
            save_best_only=True,
            save_freq=1,
        )

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks = [model_checkpoint_callback, tensorboard_callback]

        if ray_tune:
            callbacks += [TuneReporterCallback()]

        history = self.net.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
        )

        # plt.figure()
        # plt.plot(history.history['loss'], label='training loss')
        # plt.plot(history.history['val_loss'], label='validation loss')
        # plt.xlabel('epochs')
        # plt.ylabel('loss')
        # plt.legend()

        # The model weights (that are considered the best) are loaded into the model.
        # self.net.load_weights(checkpoint_filepath)
        self.net.save(save_path)

    def conv_relu_bn_dropout(
        self,
        filters,
        kernel_size=(3, 3),
        padding='same',
        bn_momentum=0.99,
        dropout_rate=0.2,
        dropout=True,
        input_shape=None,
        bn_before_relu=True,
    ):
        if input_shape is None:
            self.net.add(
                Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding=padding,
                    kernel_initializer=self.hyper_space['weight_init'],
                )
            )
        else:
            self.net.add(
                Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding=padding,
                    kernel_initializer=self.hyper_space['weight_init'],
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


class Trainer:
    def __init__(self, x_train, y_train, x_test, y_test, epochs, batch_size, save_path, ray_tune):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.epochs = epochs
        self.batch_size = batch_size
        self.save_path = save_path

        self.ray_tune = ray_tune

    def train_model(self, params):
        # x_train, x_test, labels_train, y_train, y_test, labels_test = preprocess()

        simpnet = SimpNet(params, Slim())
        simpnet.build_network_graph()
        simpnet.fit(
            self.save_path,
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
            self.epochs,
            self.batch_size,
            self.ray_tune,
        )


def main1():
    x_train, x_test, labels_train, y_train, y_test, labels_test = preprocess()

    hyperparameter_space = {
        'base_lr': tune.loguniform(0.01, 1),
        'weight_decay': tune.uniform(0.1, 0.90),
        'lr_momentum': tune.uniform(0.9, 0.99),
        'weight_init': tune.choice(['HeUniform', 'GlorotUniform', 'HeNormal', 'GlorotUniform']),
    }

    epochs = 1
    batch_size = 100
    save_path = 'SimpNet.h5'
    ray_tune = True

    trainer = Trainer(x_train, y_train, x_test, y_test, epochs, batch_size, save_path, ray_tune)

    ray.shutdown()
    ray.init(log_to_driver=False)
    analysis = tune.run(
        trainer.train_model,
        verbose=3,
        config=hyperparameter_space,
        num_samples=2,
        resources_per_trial={'cpu': 2, "gpu": 1},
    )

    logdir = analysis.get_best_logdir('keras_info/val_accuracy', mode='max')
    tuned_model = load_model(logdir + f'/{save_path}')

    tuned_loss, tuned_accuracy = tuned_model.evaluate(x_test, y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))


def main2():
    x_train, x_test, labels_train, y_train, y_test, labels_test = preprocess()

    hyperparameter_space = {
        'base_lr': 0.2,
        'decay_rate': 0.96,
        'decay_steps': 5000,
        'lr_momentum': 0.95,
        'weight_init': 'HeUniform',
    }
    # hyperparameter_space = {
    #     'base_lr': 0.1,
    #     'weight_decay': 0.7,
    #     'lr_momentum': 0.95,
    #     'weight_init': 'HeNormal',
    # }

    epochs = 80
    batch_size = 100
    save_path = 'SimpNet.h5'
    ray_tune = False

    simpnet = SimpNet(hyperparameter_space, Slim())
    simpnet.build_network_graph()
    simpnet.fit(save_path, x_train, y_train, x_test, y_test, epochs, batch_size, ray_tune)

    # logdir = analysis.get_best_logdir('keras_info/val_accuracy', mode='max')
    # tuned_model = load_model(logdir + '/model.h5')

    error_rate(save_path, x_test, labels_test)


if __name__ == "__main__":
    main2()
