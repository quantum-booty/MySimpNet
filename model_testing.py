from keras.models import load_model
from keras.utils import to_categorical, plot_model
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


def error_rate(model_path, x_test, labels_test):
    net = load_model(model_path)
    outputs = net.predict(x_test)
    labels_predicted = np.argmax(outputs, axis=1)
    misclassified = sum(labels_predicted != labels_test)
    print('pct misclassified = ', 100 * misclassified / labels_test.size)


if __name__ == "__main__":
    # (x_train, labels_train), (x_test, labels_test) = mnist.load_data()
    data = np.loadtxt(open('test.csv', 'r'), delimiter=',', skiprows=1)
    x_test = data
    # x_test = data[:, :-1]
    # labels_test = data[:, -1]

    # convert to float type
    x_test = x_test.astype('float32')
    x_test /= 255

    # # one hot encoding
    # y_test = to_categorical(labels_test, 10)

    # for i in range(10):
    #     plt.imshow(x_test[i].reshape(28, 28), cmap=plt.get_cmap('gray_r'))
    #     plt.show()

    n_pixels = 28
    x_test = x_test.reshape(x_test.shape[0], n_pixels, n_pixels, 1)

    # error_rate('models/SimpNet_30_slim_v2.h5', x_test, labels_test)
    net = load_model('SimpNet_30_slim_v2.h5')
    outputs = net.predict(x_test)
    labels_predicted = np.argmax(outputs, axis=1)
    image_ids = np.arange(1, len(x_test) + 1)
    np.savetxt(
        fname='submission.csv',
        X=np.array([image_ids, labels_predicted]).T.astype(int),
        header='ImageId,Label',
        fmt='%i',
        comments='',
        delimiter=',',
    )
