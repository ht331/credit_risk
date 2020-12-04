import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import time

a = time.time()

from prepare import readbunchobj
data = readbunchobj('dataset_delstr.data')
X_train = pd.DataFrame(data.X_train)
X_test = data.X_test
y_train = data.y_train
y_test = data.y_test

prep = StandardScaler()
X_train = prep.fit_transform(X_train)
X_test = prep.transform(X_test)

X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=10)

osp = RandomUnderSampler(random_state=10)
# osp = BorderlineSMOTE()
X_train_, y_train_ = osp.fit_sample(X_train_, y_train_)
# build model

model = Sequential()
model.add(Dense(128, input_dim=19, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
adam = Adam(lr=0.001, decay=1e-5)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[tf.keras.metrics.AUC()])
model.fit(X_train_,
          y_train_,
          epochs=10,
          batch_size=32,
          verbose=1,  # display
          validation_data=(X_val, y_val))

# score = model.evaluate(x, y)
# print(score)

y_pred = model.predict_classes(X_test, batch_size=128)
c_m = metrics.confusion_matrix(y_test, y_pred)
print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))


b = time.time()

print(b - a)