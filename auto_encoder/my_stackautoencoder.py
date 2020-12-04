# 堆叠自编码器

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


class StackedAutoEncoder(object):

    def __init__(self, list1, activation_function, eta=0.01):
        """

        :param list1: [input_dimension, hidden_layer_1, ... , hidden_layer_n]
        :param eta:
        """

        N = len(list1) - 1
        self._m = list1[0]
        self.learning_rate = eta
        self.activation_func = activation_function
        # Create the Computational graph
        self._W = {}
        self._b = {}
        self._X = {}
        self._X['0'] = tf.placeholder('float', [None, list1[0]])

        for i in range(N):
            layer = '{0}'.format(i + 1)
            print('AutoEncoder Layer {0}: {1} --> {2}'.format(layer, list1[i], list1[i + 1]))
            self._W['E' + layer] = tf.Variable(tf.random_normal(shape=(list1[i], list1[i + 1])),
                                               name='WtsEncoder' + layer)
            self._b['E' + layer] = tf.Variable(np.zeros(list1[i + 1]).astype(np.float32),
                                               name='BiasEncoder' + layer)
            self._X[layer] = tf.placeholder('float', [None, list1[i + 1]])
            self._W['D' + layer] = tf.transpose(self._W['E' + layer])  # Shared weights
            self._b['D' + layer] = tf.Variable(np.zeros(list1[i]).astype(np.float32),
                                               name='BiasDecoder' + layer)
        # Placeholder for inputs
        # self._X_noisy = tf.placeholder('float', [None, self._m])
        self.train_ops = {}
        self.out = {}

        for i in range(N):
            layer = '{0}'.format(i + 1)
            prev_layer = '{0}'.format(i)
            opt = self.pretrain(self._X[prev_layer], layer)
            self.train_ops[layer] = opt
            self.out[layer] = self.one_pass(self._X[prev_layer], self._W['E' + layer],
                                            self._b['E' + layer],
                                            self._b['D' + layer])

        self.y = self.encoder(self._X['0'], N)  # Encoder output
        self.r = self.decoder(self.y, N)  # Decoder output

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        error = self._X['0'] - self.r  # Reconstruction Error

        self._loss = tf.reduce_mean(tf.pow(error, 2))
        self._opt = optimizer.minimize(self._loss)

    def fit(self, Xtrain, layers, epochs=1, batch_size=100):
        N, D = Xtrain.shape
        num_batches = N // batch_size
        X_noisy = {}
        X = {}
        # X_noisy['0'] = Xtr_noisy
        X['0'] = Xtrain

        for i in range(layers):
            Xin = X[str(i)]
            print('Pretrainint Layer ', i + 1)
            for e in range(5):
                for j in range(num_batches):
                    batch = Xin[j * batch_size: (j * batch_size + batch_size)]
                    self.session.run(self.train_ops[str(i + 1)], feed_dict={self._X[str(i)]: batch})
            print('Pretrain Finished')
            X[str(i + 1)] = self.session.run(self.out[str(i + 1)], feed_dict={self._X[str(i)]: Xin})

        obj = []
        for i in range(epochs):
            for j in range(num_batches):
                batch = Xtrain[j * batch_size: (j * batch_size + batch_size)]
                # batch_noisy = Xtr_noisy[j * batch_size: (j * batch_size + batch_size)]
                _, ob = self.session.run([self._opt, self._loss], feed_dict={self._X['0']: batch
                                                                             })
                if j % 100 == 0:
                    print('train epoch {0} batch {2} cost {1}'.format(i, ob, j))
                obj.append(ob)
        return obj

    def encoder(self, X, N):
        x = X
        for i in range(N):
            layer = '{0}'.format(i + 1)
            hiddenE = self.activation_func(tf.matmul(x, self._W['E' + layer]) + self._b['E' + layer])
            x = hiddenE
        return x

    def decoder(self, X, N):
        x = X
        for i in range(N, 0, -1):
            layer = '{0}'.format(i)
            hiddenD = self.activation_func(tf.matmul(x, self._W['D' + layer]) + self._b['D' + layer])
            x = hiddenD
        return x

    def set_session(self, session):
        self.session = session

    def reconstruct(self, x, n_layers):
        h = self.encoder(x, n_layers)
        r = self.decoder(h, n_layers)
        return self.session.run(r, feed_dict={self._X['0']: x})

    def pretrain(self, X, layer):
        y = tf.nn.sigmoid(tf.matmul(X, self._W['E' + layer]) + self._b['E' + layer])
        r = tf.nn.sigmoid(tf.matmul(y, self._W['D' + layer]) + self._b['D' + layer])

        # Objective Function
        error = X - r  # Reconstruction Error
        loss = tf.reduce_mean(tf.pow(error, 2))
        opt = tf.train.AdamOptimizer(.001).minimize(loss,
                                                    var_list=[self._W['E' + layer], self._b['E' + layer],
                                                              self._b['D' + layer]])
        return opt

    def one_pass(self, X, W, b, c):
        h = tf.nn.sigmoid(tf.matmul(X, W) + b)
        return h


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/")
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    Xtrain = trX.astype(np.float32)
    #Xtrain_noisy = corruption(Xtrain).astype(np.float32)
    Xtest = teX.astype(np.float32)
    #Xtest_noisy = corruption(Xtest).astype(np.float32)

    _, m = Xtrain.shape

    list1 = [m, 500, 50]  # List with number of neurons in Each hidden layer, starting from input layer
    n_layers = len(list1) - 1
    autoEncoder = StackedAutoEncoder(list1, tf.nn.sigmoid)

    # Initialize all variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        autoEncoder.set_session(sess)
        err = autoEncoder.fit(Xtrain, n_layers, epochs=30)
        out = autoEncoder.reconstruct(Xtest[0: 100], n_layers)

    plt.plot(err)
    plt.xlabel('epoches')
    plt.ylabel('Fine Tuning Reconstruction Error')

    plt.show()