# 特征工程画图

import pandas as pd
import numpy as np
from prepare import readbunchobj
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def myhist(data_1, data_0, col_name, bins):

    hist_value1 = data_1[col_name]
    hist_value0 = data_0[col_name]
    group = list(np.linspace(min(hist_value1.min(), hist_value0.min()),
                             max(hist_value1.max(), hist_value0.max()),
                             bins))

    plt.hist(list(hist_value1) + list(hist_value0), group, histtype='bar', rwidth=0.8, label='label = 1')
    plt.hist(hist_value0, group, histtype='bar', rwidth=0.8, label='label = 0')
    plt.title(col_name)
    plt.legend()
    plt.savefig('%s.png' % col_name)
    plt.close()


def coef():

    data = readbunchobj('data_set.data')
    cate_feature_col = ['gender', 'receipt_address', 'household_register']

    X = data.X_train
    y = data.y_train
    col = data.col
    x_num_col = [i for i in list(col) if i not in cate_feature_col]

    # prep = StandardScaler()
    # X = prep.fit_transform(X[x_num_col])
    # X = pd.DataFrame(data=X, columns=x_num_col)
    X = X[x_num_col]

    coef_list = []
    for c in X.columns:
        coef_ = np.corrcoef(X[c], y)
        coef_list.append([c, coef_[0][1]])
    coef_df = pd.DataFrame(coef_list)


if __name__ == '__main__':
    dataset = readbunchobj('data.data')
    data = dataset.data
    label = dataset.label

    positive_index = label[label['label'] == 1].index.values
    negative_index = label[label['label'] == 0].index.values

    data_1 = data.loc[positive_index]
    data_0 = data.loc[negative_index]

    miss_df = data.isna().sum() / len(data)
    plt.barh(miss_df.index, miss_df.values)

    col = ['years', 'score', 'account_rank', 'deal_order_number',
       'avg_order_amount', 'max_pay_amount', 'last_consume_days',
       'avg_discount', 'earliest_consume_days', 'hist_consume_days',
       'order_refund_times', 'phone_number', 'application_platform_number',
       'application_number', 'apply_max_interval', 'phone_number_rank',
       'blacklist', 'receipt_phone_address_agreement', 'nan_number']

    for c in col:
        myhist(data_1, data_0, c, 20)






