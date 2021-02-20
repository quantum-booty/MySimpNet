import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


# from https://github.com/yu4u/cutout-random-erasing
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
    """
    p : the probability that random erasing is performed
    s_l, s_h : minimum / maximum proportion of erased area against input image
    r_1, r_2 : minimum / maximum aspect ratio of erased area
    v_l, v_h : minimum / maximum value for erased area
    pixel_level : pixel-level randomization for erased area
    """

    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape
        else:
            raise RuntimeError('invalid image dimensions')

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top : top + h, left : left + w] = c

        return input_img

    return eraser


class GetData:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.data_import()
        # self.preprocess()

    def get_all(self):
        return (
            self.emnist_train,
            self.emnist_test,
        )

    def data_import(self):
        # import from mnist and emnist and concatenate the two datasets
        # (x_train, labels_train), (x_test, labels_test) = mnist.load_data()

        # Construct a tf.data.Dataset
        # e_train, e_test = tfds.load('emnist/digits', split=['train', 'test'], shuffle_files=True)
        # ori_train, ori_test = tfds.load('emnist/mnist', split=['train', 'test'], shuffle_files=True)
        (emnist_digit_train, emnist_digit_test), emnist_digit_info = tfds.load(
            'emnist/digits',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        (emnist_mnist_train, emnist_mnist_test), emnist_mnist_info = tfds.load(
            'emnist/mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        (mnist_train, mnist_test), mnist_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        # print(emnist_digit_info)
        # print(emnist_mnist_info)
        # print(mnist_info)
        emnist_digit_train_len = emnist_digit_info.splits['train'].num_examples
        emnist_mnist_train_len = emnist_mnist_info.splits['train'].num_examples
        mnist_train_len = mnist_info.splits['train'].num_examples
        train_size = emnist_digit_train_len + emnist_mnist_train_len + mnist_train_len

        emnist_train = emnist_digit_train.concatenate(emnist_mnist_train)
        emnist_test = emnist_digit_test.concatenate(emnist_mnist_test)

        self.emnist_train = prepare_train(emnist_train, mnist_train, self.batch_size, train_size)
        self.emnist_test = prepare_test(emnist_test, mnist_test, self.batch_size)


def transpose(image, label):
    # return ds.map(lambda x: {'image': tf.transpose(x['image'], perm=[1, 0, 2]), 'label': x['label']})
    return tf.transpose(image, perm=[1, 0, 2]), label


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


def prepare_train(emnist_train, mnist_train, batch_size, train_size):
    # emnist needs to transposed
    emnist_train = emnist_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    emnist_train = emnist_train.map(transpose, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # mnist data has correct orientations
    mnist_train = mnist_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = emnist_train.concatenate(mnist_train)
    data = data.cache()  # cache before batch
    data = data.shuffle(train_size)
    data = data.batch(batch_size)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)
    return data


def prepare_test(emnist_test, mnist_test, batch_size):
    # emnist needs to transposed
    emnist_test = emnist_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    emnist_test = emnist_test.map(transpose, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # mnist data has correct orientations
    mnist_test = mnist_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = emnist_test.concatenate(mnist_test)
    data = data.batch(batch_size)
    data = data.cache()  # cache after batch
    data = data.prefetch(tf.data.experimental.AUTOTUNE)
    return data


def plot(x_train):
    # create a grid of 3x3 images
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_train[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()


def get_dataflow(
    x_train,
    y_train,
    batch_size,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=30,
    use_eraser=False,
    p=0.5,
    s_l=0.10,
    s_h=0.10,
    v_l=0,
    v_h=0,
    pixel_level=True,
):
    if use_eraser:
        eraser = get_random_eraser(
            p=p,
            s_l=s_l,
            s_h=s_h,
            v_l=v_l,
            v_h=v_h,
            pixel_level=pixel_level,
        )
    else:
        eraser = None
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        preprocessing_function=eraser,
    )
    datagen.fit(x_train)
    return datagen.flow(x_train, y_train, batch_size=batch_size)


if __name__ == "__main__":
    GetData()

    # x_train, x_test, labels_train, y_train, y_test, labels_test = preprocess()

    # eraser = get_random_eraser(
    #     p=0.5,
    #     s_l=0.10,
    #     s_h=0.10,
    #     v_l=0,
    #     v_h=0,
    #     pixel_level=True,
    # )
    # datagen = ImageDataGenerator(
    #     rotation_range=30,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     shear_range=30,
    #     preprocessing_function=eraser,
    # )
    # datagen.fit(x_train)

    # for x_batch, y_batch in get_dataflow(x_train, y_train, batch_size=9):
    #     plot(x_batch)
    pass
