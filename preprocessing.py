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
    def __init__(self):
        self.data_import()
        # self.preprocess()

    def get_all(self):
        return (
            self.e_train,
            self.e_test,
            self.ori_train,
            self.ori_test,
        )

    def data_import(self):
        # import from mnist and emnist and concatenate the two datasets
        # (x_train, labels_train), (x_test, labels_test) = mnist.load_data()

        def transpose(image, label):
            # return ds.map(lambda x: {'image': tf.transpose(x['image'], perm=[1, 0, 2]), 'label': x['label']})
            return tf.transpose(image, perm=[1, 0, 2]), label

        def normalize_img(image, label):
            return tf.cast(image, tf.float32) / 255.0, label

        # Construct a tf.data.Dataset
        # e_train, e_test = tfds.load('emnist/digits', split=['train', 'test'], shuffle_files=True)
        # ori_train, ori_test = tfds.load('emnist/mnist', split=['train', 'test'], shuffle_files=True)
        (e_train, e_test), e_info = tfds.load(
            'emnist/digits',
            split=['train[:10]', 'test[:10]'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        (ori_train, ori_test), ori_info = tfds.load(
            'emnist/mnist',
            split=['train[:10]', 'test[:10]'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        e_train = e_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        e_train = e_train.map(transpose, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        e_train = e_train.cache()  # cache before batch
        e_train = e_train.shuffle(e_info.splits['train'].num_examples)
        e_train = e_train.batch(128)
        e_train = e_train.prefetch(tf.data.experimental.AUTOTUNE)

        e_test = e_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        e_test = e_test.map(transpose, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        e_test = e_test.batch(128)
        e_test = e_test.cache()  # cache after batch
        e_test = e_test.prefetch(tf.data.experimental.AUTOTUNE)

        ori_train = ori_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ori_test = ori_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.e_train = e_train
        self.e_test = e_test
        self.ori_train = ori_train
        self.ori_test = ori_test


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
