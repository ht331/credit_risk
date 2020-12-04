# 数据预处理
import pandas as pd
import numpy as np
import bunch
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


def data_pre_process():

    data = pd.read_excel('data1.xlsx')
    data = pd.get_dummies(data)
    col = data.columns
    imp = SimpleImputer(missing_values=0, strategy='mean')
    x = np.array(data['分数']).reshape(1, -1)
    imp.fit(x)
    data['分数'] = pd.DataFrame(imp.transform(x).T)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(data)
    data = pd.DataFrame(imp.transform(data))
    data.columns = col
    return data


def data_pre_process_1():
    data = pd.read_excel('data1.xlsx')
    df = data[data['num_na'] != 15]
    df = pd.get_dummies(df)
    d = df
    col = df.columns
    imp_0 = SimpleImputer(missing_values=0, strategy='mean')
    imp_na = SimpleImputer(missing_values=np.nan, strategy='mean')
    f = np.array(d['分数']).reshape(1, -1)
    imp_0.fit(f)
    d['分数'] = pd.DataFrame(imp_0.transform(f).T)
    imp_na.fit(d)
    c_data = pd.DataFrame(imp_na.transform(d))
    c_data.columns = col
    col = list(col)
    y_col = ['逾期标签']
    x_col = [i for i in col if i not in ['逾期标签', 'ID']]

    c_data[x_col].to_excel('X.xlsx', index=False)
    c_data[y_col].to_excel('y.xlsx', index=False)
    return c_data


def predata():

    data = pd.read_excel('data1.xlsx')
    df = data[data['num_na'] != 15]
    df = pd.get_dummies(df)
    df.fillna(-1, inplace=True)
    col = list(df.columns)
    y_col = ['逾期标签']
    x_col = [i for i in col if i not in ['逾期标签', 'ID']]
    df[x_col].to_excel('X1.xlsx', index=False)
    df[y_col].to_excel('y1.xlsx', index=False)


def Knn_pre():

    data = pd.read_excel('data1.xlsx')
    df = data[data['num_na'] != 15]
    df = pd.get_dummies(df)
    col = list(df.columns)
    df['分数'].replace(0, np.nan, inplace=True)
    y_col = ['逾期标签']
    x_col = [i for i in col if i not in ['逾期标签', 'ID']]

    X = df[x_col[:19]]
    y = df[y_col]
    imp = KNNImputer()
    XX = imp.fit_transform(X)


def prepare():
    """
        对字符串进行哑编码
    :return:
    """
    data = pd.read_excel("data1.xlsx")
    df = data[data['num_na'] != 15]
    df = pd.get_dummies(df)
    col = list(df.columns)
    df['分数'].replace(0, np.nan, inplace=True)
    y_col = ['逾期标签']
    x_col = [i for i in col if i not in ['逾期标签', 'ID']]
    X = df[x_col]
    X['逾期标签'] = df[y_col]
    X.to_excel('data_set.xlsx', index=False)


def prepare_catboost():

    col = ['years', 'score', 'account_rank', 'deal_order_number',
            'avg_order_amount', 'max_pay_amount', 'last_consume_days', 'avg_discount', 'earliest_consume_days',
            'hist_consume_days', 'order_refund_times', 'phone_number', 'application_platform_number',
            'application_number', 'apply_max_interval', 'phone_number_rank', 'blacklist',
            'receipt_phone_address_agreement', 'nan_number', 'gender', 'receipt_address', 'household_register']
    cate_feature_col = ['gender', 'receipt_address', 'household_register']

    dataset = readbunchobj('data.data')

    data = dataset.data
    label = dataset.label

    data['label'] = label
    data = data[data['nan_number'] < 10]

    X = data[col]
    y = pd.DataFrame(data['label'])

    col_del = ['phone_number', 'application_platform_number', 'application_number', 'apply_max_interval']
    for c in col_del:
        del X[c]

    x_num_col = [i for i in list(X.columns) if i not in cate_feature_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    y_train = list(y_train['label'])
    y_test = list(y_test['label'])
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    imp = SimpleImputer(strategy='mean')  # 均值 单变量插补

    X1 = imp.fit_transform(X_train[x_num_col])
    X0 = imp.transform(X_test[x_num_col])

    prep = StandardScaler()
    X1 = prep.fit_transform(X1)
    X0 = prep.transform(X0)

    pca = PCA(n_components=0.99, svd_solver='full')
    X11 = pca.fit_transform(X1)
    X00 = pca.transform(X0)

    X_train_ = pd.DataFrame(np.hstack((X1, X_train[cate_feature_col])))
    X_test_ = pd.DataFrame(np.hstack((X0, X_test[cate_feature_col])))

    X_train_[X_train_.columns[:-3]] = X_train_[X_train_.columns[:-3]].astype(float)
    X_test_[X_test_.columns[:-3]] = X_test_[X_test_.columns[:-3]].astype(float)

    X_train_[X_train_.columns[-3:]] = X_train_[X_train_.columns[-3:]].astype(str)
    X_test_[X_test_.columns[-3:]] = X_test_[X_test_.columns[-3:]].astype(str)

    dataset = bunch.Bunch(X_train=X_train_,
                          y_train=y_train,
                          X_test=X_test_,
                          y_test=y_test,
                          col=col)
    writeBunchobj('dataset.data', dataset)


def prepare_delstr():

    data = pd.read_excel('data1.xlsx')
    # data = data[data['num_na'] < 15]

    data['分数'].replace(0, np.nan, inplace=True)

    col = data.columns
    x_col = [i for i in col if i not in ['逾期标签', 'ID']]
    y_col = ['逾期标签']

    cate_feature_col = ['性别', '收货地址', '户籍']
    x_num_col = [i for i in x_col if i not in cate_feature_col]
    X = data[x_num_col]
    y = data[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    y_train = list(y_train['逾期标签'])
    y_test = list(y_test['逾期标签'])
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    imp = SimpleImputer(strategy='mean')  # 均值 单变量插补
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    en_col = ['years', 'score', 'account_rank', 'deal_order_number',
              'avg_order_amount', 'max_pay_amount', 'last_consume_days', 'avg_discount', 'earliest_consume_days',
              'hist_consume_days', 'order_refund_times', 'phone_number', 'application_platform_number',
              'application_number', 'apply_max_interval', 'phone_number_rank', 'blacklist',
              'receipt_phone_address_agreement', 'nan_number',]
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    dataset = bunch.Bunch(X_train=X_train,
                          y_train=y_train,
                          X_test=X_test,
                          y_test=y_test,
                          col=en_col)
    writeBunchobj('dataset_delstr.data', dataset)


def readbunchobj(path):
    file_obj = open(path, 'rb')
    bunch = pickle.load(file_obj)
    file_obj.close()
    return bunch


def writeBunchobj(path, bunchobj):
    file_obj = open(path, 'wb')
    pickle.dump(bunchobj, file_obj)
    file_obj.close()


def data_prepare():

    col = ['years', 'score', 'account_rank', 'deal_order_number',
            'avg_order_amount', 'max_pay_amount', 'last_consume_days', 'avg_discount', 'earliest_consume_days',
            'hist_consume_days', 'order_refund_times', 'phone_number', 'application_platform_number',
            'application_number', 'apply_max_interval', 'phone_number_rank', 'blacklist',
            'receipt_phone_address_agreement', 'nan_number', 'gender', 'receipt_address', 'household_register']

    cate_feature_col = ['gender', 'receipt_address', 'household_register']

    dataset = readbunchobj('data.data')

    data = dataset.data
    label = dataset.label

    data['label'] = label
    data = data[data['nan_number'] < 15]
    X = data[col]
    y = pd.DataFrame(data['label'])
    x_num_col = [i for i in list(X.columns) if i not in cate_feature_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    y_train = list(y_train['label'])
    y_test = list(y_test['label'])
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    imp_mean = SimpleImputer(strategy='mean')  # 均值 单变量插补
    imp_mode = SimpleImputer(strategy='median')

    mean_col = ['score', 'avg_discount']
    mode_col = ['account_rank', 'deal_order_number', 'avg_order_amount', 'max_pay_amount', 'last_consume_days',
                'earliest_consume_days', 'hist_consume_days', 'order_refund_times', 'phone_number',
                'application_platform_number', 'application_number', 'apply_max_interval', 'phone_number_rank',
                ]

    X_train_ = X_train.copy()
    X_test_ = X_test.copy()
    for c in mean_col:
        X_train_[c] = imp_mean.fit_transform(np.array(X_train[c]).reshape(-1, 1))
        X_test_[c] = imp_mean.transform(np.array(X_test[c]).reshape(-1, 1))

    for c in mode_col:
        X_train_[c] = imp_mode.fit_transform(np.array(X_train[c]).reshape(-1, 1))
        X_test_[c] = imp_mode.transform(np.array(X_test[c]).reshape(-1, 1))

    dataset = bunch.Bunch(X_train=X_train_,
                          y_train=y_train,
                          X_test=X_test_,
                          y_test=y_test,
                          col=col)
    writeBunchobj('data_set.data', dataset)


if __name__ == "__main__":
    prepare_delstr()
