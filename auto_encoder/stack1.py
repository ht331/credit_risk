
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data')
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
Xtrain = trX.astype(np.float32)
Xtest = teX.astype(np.float32)
_, m = Xtrain.shape


learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784

X = tf.placeholder('float', [None, n_input])

weights = {'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
           'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
           'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
           'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))}

biases = {'encoder_h1': tf.Variable(tf.random_normal([n_hidden_1])),
          'encoder_h2': tf.Variable(tf.random_normal([n_hidden_2])),
          'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1])),
          'decoder_h2': tf.Variable(tf.random_normal([n_input]))}


def encoder(x, activation_func):
    layer1 = activation_func(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_h1']))
    layer2 = activation_func(tf.add(tf.matmul(layer1, weights['encoder_h2']), biases['encoder_h2']))
    return layer2


def decoder(x, activation_func):
    layer1 = activation_func(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_h1']))
    layer2 = activation_func(tf.add(tf.matmul(layer1, weights['decoder_h2']), biases['decoder_h2']))
    return layer2


encoder_op = encoder(X, tf.nn.sigmoid)
decoder_op = decoder(encoder_op, tf.nn.sigmoid)
y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    N, D = Xtrain.shape
    num_batches = N // batch_size
    obj = []
    for i in range(training_epochs):
        # X = shuffle(X)
        for j in range(num_batches):
            batch = Xtrain[j * batch_size: (j * batch_size + batch_size)]
            _, ob = sess.run([optimizer, cost], feed_dict={X: batch})
            if j % 100 == 0 and i % 100 == 0:
                print('training epoch {0} batch {2} cost {1}'.format(i, ob, j))
            obj.append(ob)


