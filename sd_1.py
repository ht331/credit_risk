#!/usr/bin/python
# -*- coding: UTF-8 -*-

from keras.datasets import imdb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from keras import models
# from keras import layers
import matplotlib.pyplot as plt
from prepare import readbunchobj
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn import metrics
from tensorflow.keras.layers import Layer


# 构建数据
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=19):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)
#
# y_train = np.asarray(train_labels).astype('float32')
# y_test = np.asarray(test_labels).astype('float32')

data = readbunchobj('d:/py/credit_risk/dataset_delstr.data')
Xtrain = np.array(data.X_train)
Xtest = data.X_test
y_train = data.y_train
y_test = data.y_test
prep = MinMaxScaler()
x_train = prep.fit_transform(Xtrain)
x_test = prep.transform(Xtest)

# osp = RandomUnderSampler(random_state=10)
# osp = RandomOverSampler(random_state=10)
osp = SMOTE()
x_train, y_train = osp.fit_sample(x_train, y_train)

# 构建网络
n, m = x_train.shape
model = keras.models.Sequential()
model.add(layers.Dense(18, activation='relu', input_shape=(19,)))
model.add(layers.Dense(15, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`


# def vae_loss(x, x_decoded_mean):
#     xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
#     kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
#     return xent_loss + kl_loss


# vae.compile(optimizer='rmsprop', loss=vae_loss)


# 构建优化算法和损失算法
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


x_val = x_train[:5000]
partial_x_train = x_train[5000:]
y_val = y_train[:5000]
partial_y_train = y_train[5000:]



# 训练模型
history = model.fit(partial_x_train, partial_y_train, epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

y_pred = model.predict_classes(x_test)
c_m = metrics.confusion_matrix(y_test, y_pred)
print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))

# 显示训练数据
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
