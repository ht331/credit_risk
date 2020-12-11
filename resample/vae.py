# 变分自编码器
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
import keras.backend as K
import numpy as np

original_dim = 19
intermediate_dim = 15
latent_dim = 10

inputs = Input(shape=(original_dim, ))
h = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), 
                              mean=0, stddev=0.1)

    return z_mean + K.exp(z_log_sigma) * epsilon


def get_loss(args):
    """
    自定义损失函数
    x:原始样本
    xr: 重构样本
    m: 编码器隐变量z均值
    v: 编码器隐变量z方差的对数
    """
    x, xr, m, v = args
    dim = K.int_shape(x)[-1]
    re_loss = dim * binary_crossentropy(x, xr)  # 重构正确性度量
    kl_loss = 1 + v - K.square(m) - K.exp(v)  # d维向量，隐变量z维度为d
    kl_loss = - 0.5 * K.sum(kl_loss, axis=-1)  # kl散度，罚项
    vae_loss = K.mean(re_loss + kl_loss)
    return vae_loss

z = Lambda(sampling)([z_mean, z_log_sigma])

# Create encoder
encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Create decoder
latent_inputs = Input(shape=(latent_dim, ), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)
decoder = Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
x_decoded = decoder(encoder(inputs)[2])
outputs = Lambda(get_loss)([inputs, x_decoded, z_mean, z_log_sigma])  # 模型直接输出损失函数值
vae = Model(inputs, outputs, name='vae_mlp')

# reconstruction_loss = binary_crossentropy(inputs, outputs)
# reconstruction_loss *= original_dim
# kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
# kl_loss = K.sum(kl_loss, axis=-1)
# kl_loss *= -0.5
# vae_loss = K.mean(reconstruction_loss + kl_loss)
# vae.add_loss(vae_loss)
vae.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)


# train data
import pandas as pd
from prepare import readbunchobj
from sklearn.preprocessing import MinMaxScaler
data = readbunchobj('dataset_delstr.data')
x_train = np.array(data.X_train)
x_test = np.array(data.X_test)
y_train = data.y_train
y_test = data.y_test

scl = MinMaxScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

train = pd.DataFrame(x_train)
train['target'] = y_train
train_0 = train[train['target'] == 0]  # 多数类
train_1 = train[train['target'] == 1]  # 少数类

resample_num = 2000  # 需要生成的少数类样本数目
rs_train_x = train_1.iloc[:, :-1]
vae.fit(x_train, x_train, epochs=10, batch_size=128, shuffle=True)  # 训练vae生成模型


# f = np.load(r'D:/py/credit_risk/resample/mnist.npz')
# x_train, y_train = f['x_train'], f['y_train']
# x_test, y_test = f['x_test'], f['y_test']
# f.close()
# x_train = np.reshape(x_train, [-1, original_dim])
# x_test = np.reshape(x_test, [-1, original_dim])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
# vae.fit(x_train, x_train, shuffle=True, epochs=10, batch_size=128)

# resample_data = pd.DataFrame()
# for i in range(int(resample_num / 100)):
#     samp = rs_train_x.sample(n=100)  # 随机抽取100个样本生成
#     ei = encoder.predict(samp)[2]
#     di = decoder.predict(ei)
#     resample_data = resample_data.append(pd.DataFrame(di))

# return_data = resample_data
# return_data = rs_train_x
x_train_des = encoder.predict(x_train)[2]
x_test_des = encoder.predict(x_test)[2]

from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule, TomekLinks

# X_train = pd.DataFrame()
# all_train=return_data.copy()
# all_train['target'] = 1
# all_train = all_train.append(train_0.sample(n=len(rs_train_x)))
# X_train = all_train.iloc[:, :-1]
# y_train = all_train['target']
# osp = SMOTE(random_state=10)
osp = RandomUnderSampler(random_state=10)
X_train, y_train = osp.fit_sample(x_train_des, y_train)  # SMOTE

clf = AdaBoostClassifier(n_estimators=100, random_state=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(x_test_des)
# y_pred = my_adaboost_clf(y_train, X_train, y_test, X_test, M=100)

c_m = metrics.confusion_matrix(y_test, y_pred)
print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))