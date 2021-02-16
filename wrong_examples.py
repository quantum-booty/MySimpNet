from keras.models import load_model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
net = load_model('SimpNet_30_slim_v2.h5')
outputs = net.predict(x_test)
labels_predicted = np.argmax(outputs, axis=1)
misclassified = sum(labels_predicted != labels_test)
print('pct misclassified = ', 100 * misclassified / labels_test.size)

wrong_idx = labels_predicted != labels_test
x_wrong = x_test[wrong_idx]
correct_labels_wrong = labels_test[wrong_idx]

for i in np.random.choice(np.arange(len(x_wrong)), size=20, replace=False):
    print(correct_labels_wrong[i])
    plt.imshow(x_wrong[i].reshape(28, 28), cmap=plt.get_cmap('gray_r'))
    plt.show()
