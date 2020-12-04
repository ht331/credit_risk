
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prepare import readbunchobj


class StackedAutoEncoder(object):

    def __init__(self, activation_func, list1, eta):

        self.N = len(list1) - 1
        self._m = list1[0]
        self.learning_rate = eta
        self.activation_func = activation_func
        # Create the Computational graph
        self._W = {}
        self._b = {}
        # self._X = {}
        # self._X['0'] = tf.placeholder('float', [None, list1[0]])
        self._X = tf.compat.v1.placeholder('float', [None, self._m])

        for i in range(self.N):
            layer = '{0}'.format(i + 1)
            print('AutoEncoder Layer {0}: {1} --> {2}'.format(layer, list1[i], list1[i + 1]))
            self._W['E' + layer] = tf.Variable(tf.random.normal([list1[i], list1[i + 1]]))
            self._b['E' + layer] = tf.Variable(np.zeros(list1[i + 1]).astype(np.float32))
            # self._X[layer] = tf.placeholder('float', [None, list1[i + 1]])
            # self._W['D' + layer] = tf.transpose(self._W['E' + layer])  # Shared weights
            self._W['D' + layer] = tf.Variable(tf.random.normal([list1[self.N - i], list1[self.N - i - 1]]))
            self._b['D' + layer] = tf.Variable(np.zeros(list1[self.N - i - 1]).astype(np.float32))

        self._encoder_op = self.encoder(self._X)
        self._decoder_op = self.decoder(self._encoder_op)
        error = self._X - self._decoder_op

        self._cost = tf.reduce_mean(tf.pow(error, 2))
        self._optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self._cost)

    def encoder(self, X):
        x = X
        for i in range(self.N):
            layer = '{0}'.format(i + 1)
            hidden_encoder = self.activation_func(tf.matmul(x, self._W['E' + layer]) + self._b['E' + layer])
            x = hidden_encoder
        return x

    def decoder(self, X):
        x = X
        for i in range(self.N):
            layer = '{0}'.format(i + 1)
            hidden_decoder = self.activation_func(tf.matmul(x, self._W['D' + layer]) + self._b['D' + layer])
            x = hidden_decoder
        return x

    def set_session(self, session):
        self.sess = session

    def fit(self, xtrain,  training_epochs, batch_size=100):
        N, D = xtrain.shape
        num_batches = N // batch_size
        for i in range(training_epochs):
            # X = shuffle(X)
            for j in range(num_batches):
                batch = xtrain[j * batch_size: (j * batch_size + batch_size)]
                _, ob = self.sess.run([self._optimizer, self._cost], feed_dict={self._X: batch})
                if j % 100 == 0:
                    print('training epoch {0} batch {2} cost {1}'.format(i, ob, j))

    def transform(self, x):
        return self.sess.run(self._encoder_op, feed_dict={self._X: x})


class DualStackedAutoEncoder(object):

    def __init__(self, list1, eta):

        self.stack_auto_encoder_sigmoid = StackedAutoEncoder(tf.nn.sigmoid, list1, eta)
        self.stack_auto_encoder_tanh = StackedAutoEncoder(tf.nn.tanh, list1, eta)

    def fit_transform(self, xtrain, xtest, training_epochs, bitch_size):
        init = tf.compat.v1.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(init)
        self.stack_auto_encoder_sigmoid.set_session(sess)
        self.stack_auto_encoder_sigmoid.fit(xtrain, training_epochs, bitch_size)
        x_train_transform_sigmoid = self.stack_auto_encoder_sigmoid.transform(xtrain)
        x_test_transform_sigmoid = self.stack_auto_encoder_sigmoid.transform(xtest)

        sess.run(init)
        self.stack_auto_encoder_tanh.set_session(sess)
        self.stack_auto_encoder_tanh.fit(xtrain, training_epochs, bitch_size)
        x_train_transform_tanh = self.stack_auto_encoder_tanh.transform(xtrain)
        x_test_transform_tanh = self.stack_auto_encoder_tanh.transform(xtest)

        sess.close()

        x_train_transform = np.hstack((x_train_transform_sigmoid, x_train_transform_tanh))
        x_test_transform = np.hstack((x_test_transform_sigmoid, x_test_transform_tanh))

        return x_train_transform, x_test_transform


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True  # allocate dynamically

    data = readbunchobj('d:/py/credit_risk/dataset_delstr.data')
    Xtrain = np.array(data.X_train)
    Xtest = data.X_test
    y_train = data.y_train
    y_test = data.y_test
    prep = MinMaxScaler()
    Xtrain = prep.fit_transform(Xtrain)
    Xtest = prep.transform(Xtest)

    n, m = Xtrain.shape
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    list1 = [m, 21, 17, 9]
    eta = 0.01
    training_epochs = 30
    bitch_size = 100

    dual_encoder = DualStackedAutoEncoder(list1, eta)

    X_train, X_test = dual_encoder.fit_transform(Xtrain, Xtest, training_epochs, bitch_size)

    from imblearn.under_sampling import RandomUnderSampler
    # from imblearn.over_sampling import RandomOverSampler
    from catboost import CatBoostClassifier, CatBoost, Pool
    from sklearn import metrics

    osp = RandomUnderSampler(random_state=10)
    # osp = RandomOverSampler(random_state=10)
    X_train, y_train = osp.fit_sample(X_train, y_train)
    clf = CatBoostClassifier(loss_function='Logloss',
                             logging_level='Silent',
                             random_state=10,
                             )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    c_m = metrics.confusion_matrix(y_test, y_pred)
    print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
    print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
    print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
    print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
    print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
    print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))

