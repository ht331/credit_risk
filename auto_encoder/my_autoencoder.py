# 标准自编码

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


class AutoEncoder(object):

    def __init__(self, m, n, eta=0.01):
        """

        :param m: number of neurons in input/output layer
        :param n: number of neurons in hidden layer
        :param eta:
        """
        self._m = m
        self._n = n
        self.learning_rate = eta

        # Weighs and biases
        self._W1 = tf.Variable(tf.random.normal(shape=(self._m, self._n)))
        self._W2 = tf.Variable(tf.random.normal(shape=(self._n, self._m)))
        self._b1 = tf.Variable(np.zeros(self._n).astype(np.float32))  # bias for hidden layer
        self._b2 = tf.Variable(np.zeros(self._m).astype(np.float32))  # bias for output layer

        # Placeholder for inputs
        self._X = tf.compat.v1.placeholder('float', [None, self._m])
        self.y = self.encoder(self._X)
        self.r = self.decoder(self.y)
        error = self._X - self.r

        self._loss = tf.reduce_mean(tf.pow(error, 2))
        self._opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self._loss)

    def encoder(self, x):
        h = tf.matmul(x, self._W1) + self._b1
        return tf.nn.sigmoid(h)

    def decoder(self, x):
        h = tf.matmul(x, self._W2) + self._b2
        return tf.nn.sigmoid(h)

    def set_session(self, session):
        self.session = session

    def reduced_dimension(self, x):
        h = self.encoder(x)
        return self.session.run(h, feed_dict={self._X: x})

    def reconstruct(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return self.session.run(r, feed_dict={self._X: x})

    def fit(self, X, epochs=1, batch_size=100):
        N, D = X.shape
        num_batches = N // batch_size

        obj = []
        for i in range(epochs):
            # X = shuffle(X)
            for j in range(num_batches):
                batch = X[j * batch_size: (j * batch_size + batch_size)]
                _, ob = self.session.run([self._opt, self._loss], feed_dict={self._X: batch})
                if j % 100 == 0 and i % 100 == 0:
                    print('training epoch {0} batch {2} cost {1}'.format(i, ob, j))
                obj.append(ob)
        return obj


if __name__ == "__main__":

    mnist = input_data.read_data_sets("MNIST_data/")
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    Xtrain = trX.astype(np.float32)
    Xtest = teX.astype(np.float32)
    _, m = Xtrain.shape

    autoEncoder = AutoEncoder(m, 256)

    # Initialize all variables
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        autoEncoder.set_session(sess)
        err = autoEncoder.fit(Xtrain, epochs=10)
        out = autoEncoder.reconstruct(Xtest[0:100])

    plt.plot(err)
    plt.xlabel('epochs')
    plt.ylabel('cost')

    # Plotting original and reconstructed images
    row, col = 2, 8
    idx = np.random.randint(0, 100, row * col // 2)
    f, axarr = plt.subplots(row, col, sharex=True, sharey=True, figsize=(20, 4))
    for fig, row in zip([Xtest, out], axarr):
        for i, ax in zip(idx, row):
            ax.imshow(fig[i].reshape((28, 28)), cmap='Greys_r')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()