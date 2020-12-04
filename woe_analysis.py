
import numpy as np
import pandas as pd
from prepare import readbunchobj
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
from prepare import writeBunchobj
import bunch


def get_data():
    data = readbunchobj('data.data')
    return data


def count_binary(a, event):
    event_count = (a == event).sum().values[0]
    non_event_count = len(a) - event_count
    return event_count, non_event_count


def woe_iv_value(a, total_good, total_bad):
    bin_good, bin_bad = count_binary(a, 0)
    odds = bin_bad / bin_good
    mar_g_rate = bin_good / total_good
    mar_b_rate = bin_bad / total_bad
    if bin_good == 0 or bin_bad == 0:
        woe_value = np.log(((bin_bad + 0.5) / (bin_good + 0.5)) / (total_bad / total_good))
    else:
        woe_value = np.log(mar_b_rate / mar_g_rate)
    iv_value = ((bin_bad / total_bad) - (bin_good / total_good)) * woe_value
    return len(a), bin_good, bin_bad, mar_g_rate, mar_b_rate, odds, woe_value, iv_value


def make_group(x, k):

    no_nan_x = x[x.isna() == False]  # 去掉nan值的其他值
    value_list = list(set(list(no_nan_x)))  # 唯一值
    if (len(value_list) > k) and (type(value_list[0]) != str):
        # 进行分箱
        # group = list(np.linspace(x.min(), x.max(), 10))
        group = list(x.quantile([1.0 * i / k for i in range(k + 1)], interpolation="lower").drop_duplicates(
            keep="last"))

    elif type(value_list[0]) != str:
        value_list.sort()
        group = value_list
    else:
        group = value_list
    return group


def woe(x, y, group, group_type):

    no_nan_x = x[x.isna() == False]  # 去掉nan值的其他值
    nan_x = x[x.isna()]  # nan值

    value_list = list(set(list(no_nan_x)))  # 唯一值

    total_bad = y.sum().values[0]
    total_good = len(y) - total_bad

    return_value = []
    if group_type == 'cate':
        # 类型特征
        for v in group:
            a = y.loc[no_nan_x[no_nan_x == v].index]
            obs, bin_good, bin_bad, mar_g_rate, mar_b_rate, odds, woe_value, iv_value \
                = woe_iv_value(a, total_good, total_bad)
            return_value.append([v, v, v, obs, bin_good, bin_bad, mar_g_rate, mar_b_rate, odds, woe_value, iv_value])

    else:

        for i in range(len(group) - 1):
            if i == 0:
                a = y.loc[no_nan_x[no_nan_x <= group[1]].index]
                min_ = 0
                max_ = group[1]
            elif i == len(group) - 1:
                a = y.loc[no_nan_x[no_nan_x > group[-2]].index]
                min_ = group[-2]
                max_ = 'inf'
            else:
                a = y.loc[no_nan_x[(no_nan_x > group[i]) & (no_nan_x <= group[i + 1])].index]
                min_ = group[i]
                max_ = group[i + 1]

            obs, bin_good, bin_bad, mar_g_rate, mar_b_rate, odds, woe_value, iv_value \
                = woe_iv_value(a, total_good, total_bad)
            return_value.append(
                [i, min_, max_, obs, bin_good, bin_bad, mar_g_rate, mar_b_rate, odds, woe_value, iv_value])

    if len(nan_x) != 0:
        a = y.loc[nan_x.index]
        obs, bin_good, bin_bad, mar_g_rate, mar_b_rate, odds, woe_value, iv_value \
            = woe_iv_value(a, total_good, total_bad)
        return_value.append(
            ['nan', 'nan', 'nan', obs, bin_good, bin_bad, mar_g_rate, mar_b_rate, odds, woe_value, iv_value])

    columns = ['bucket', 'min', 'max', 'obs', 'good', 'bad', 'margin_good', 'margin_bad', 'odds', 'woe', 'iv']

    return pd.DataFrame(data=return_value, columns=columns)


def plot_woe(woe_df):
    plt.plot([i for i in range(len(woe_df))], woe_df['woe'])


def one_woe(x, y, min_value, max_value):
    no_nan_x = x[x.isna() == False]  # 去掉nan值的其他值
    total_bad = y.sum().values[0]
    total_good = len(y) - total_bad
    a = y.loc[no_nan_x[(no_nan_x > min_value) & (no_nan_x <= max_value)].index]
    obs, bin_good, bin_bad, mar_g_rate, mar_b_rate, odds, woe_value, iv_value \
                    = woe_iv_value(a, total_good, total_bad)
    return woe_value


def load_feature_group():
    with open('feature_group_.json', 'r') as f:
        a = f.readline()
    return json.loads(a)


def woe_fit(X, y):

    feature_group = load_feature_group()
    feature_list = feature_group['feature']
    group_list = feature_group['group']
    type_list = feature_group['type']

    x_ = X[feature_list]

    woe_list = []
    for i in range(len(feature_list)):
        feature = feature_list[i]
        group = group_list[i]
        cate_type = type_list[i]
        feature_x = x_[feature]
        rdf = woe(feature_x, y, group, cate_type)
        woe_df = rdf[['min', 'max', 'woe']]
        woe_list.append(woe_df)
    return woe_list


def woe_transform(X, woe_list):
    feature_group = load_feature_group()
    feature_list = feature_group['feature']
    type_list = feature_group['type']
    x_test = X[feature_list].copy()

    for k in range(len(feature_list)):
        woe_df = woe_list[k]
        feature = feature_list[k]
        cate_type = type_list[k]
        if cate_type == 'cate':
            for i in range(len(woe_df)):
                woe_value = woe_df['woe'][i]
                min_ = woe_df['min'][i]
                if (i == len(woe_df) - 1) and min_ == 'nan':
                    x_test.loc[x_test[feature].isna(), feature] = woe_value
                else:
                    x_test.loc[x_test[feature] == min_, feature] = woe_value

        else:

            for i in range(len(woe_df)):
                woe_value = woe_df['woe'][i]
                min_ = woe_df['min'][i]
                max_ = woe_df['max'][i]
                if (i == len(woe_df) - 1) and min_ == 'nan':
                    x_test.loc[x_test[feature].isna(), feature] = woe_value
                elif (i == len(woe_df) - 1) and min_ != 'nan':
                    x_test.loc[x_test[feature] > min_, feature] = woe_value
                else:
                    x_test.loc[(x_test[feature] <= max_) & (x_test[feature] > min_), feature] = woe_value

    x_test.reset_index(drop=True, inplace=True)
    return x_test


def woe_transform_train(X_train, woe_list):
    return woe_transform(X_train, woe_list)


def woe_transform_test(X_test, woe_list):
    return woe_transform(X_test, woe_list)


def make_feature_group_dict():
    featurelist = ['account_rank', 'deal_order_number', 'avg_order_amount', 'max_pay_amount',
                   'last_consume_days', 'avg_discount', 'earliest_consume_days', 'hist_consume_days',
                   'order_refund_times', 'phone_number', 'application_platform_number', 'application_number',
                   'apply_max_interval', 'phone_number_rank', 'blacklist', 'receipt_phone_address_agreement', 'gender',
                   'receipt_address', 'household_register']
    grouplist = [
                 [0, 1, 3, 6],
                 [0, 1, 2, 3, 9, 13, 273],
                 [0, 46, 91, 140, 170, 2000],
                 [0, 269, 364, 499, 717, 10000],
                 [0, 18, 46, 77, 86, 142, 301, 437, 1127],
                 [0, 0.17, 0.45, 0.52, 0.69, 1],
                 [0, 774, 933, 1055, 1102, 1127],
                 [0, 3, 4, 5, 13, 205],
                 [0.0, 1.0, 2.0, 88.0],
                 [0, 1, 3],
                 [1.0, 2.0, 3.0, 29.0],
                 [1.0, 2.0, 3.0, 4.0, 15.0],
                 [0.0, 1.0, 42.0, 101.0, 160.0, 601.0],
                 [0, 4, 5, 6],
                 [0, 1],
                 [0, 1],
                 ['女', '男'],
                 ['甘肃省',
                  '河北省',
                  '青海省',
                  '内蒙古自治区',
                  '重庆',
                  '贵州省',
                  '浙江省',
                  '黑龙江省',
                  '海南省',
                  '福建省',
                  '山西省',
                  '宁夏回族自治区',
                  '湖南省',
                  '新疆维吾尔自治区',
                  '江苏省',
                  '上海',
                  '西藏自治区',
                  '广东省',
                  '陕西省',
                  '北京',
                  '湖北省',
                  '安徽省',
                  '天津',
                  '云南省',
                  '辽宁省',
                  '河南省',
                  '吉林省',
                  '山东省',
                  '广西壮族自治区',
                  '江西省',
                  '四川省'],
                 ['西藏',
                  '宁夏',
                  '浙江',
                  '重庆',
                  '安徽',
                  '山西',
                  '海南',
                  '黑龙江',
                  '内蒙古',
                  '福建',
                  '陕西',
                  '吉林',
                  '河北',
                  '广东',
                  '江西',
                  '云南',
                  '上海',
                  '北京',
                  '广西',
                  '湖南',
                  '河南',
                  '天津',
                  '山东',
                  '湖北',
                  '贵州',
                  '青海',
                  '江苏',
                  '四川',
                  '新疆',
                  '甘肃',
                  '辽宁']]
    catelist = ['blacklist', 'receipt_phone_address_agreement', 'gender', 'receipt_address', 'household_register']
    typelist = []
    for i in range(len(featurelist)):
        f = featurelist[i]
        if f in catelist:
            typelist.append('cate')
        else:
            typelist.append('no_cate')

    f_g = {'feature': featurelist, 'group': grouplist, 'type': typelist}
    f_g_str = json.dumps(f_g)
    with open('feature_group_all.json', 'w') as f:
        f.write(f_g_str)


def make_feature_group_003():
    # 取iv>0.03的

    featurelist = ['deal_order_number', 'last_consume_days', 'hist_consume_days',
                   'application_platform_number', 'application_number', 'apply_max_interval',
                   'blacklist']
    grouplist = [[0, 1, 2, 3, 9, 13, 273],
                 [0, 18, 46, 77, 86, 142, 301, 437, 1127],
                 [0, 3, 4, 5, 13, 205],
                 [1.0, 2.0, 3.0, 29.0],
                 [1.0, 2.0, 3.0, 4.0, 15.0],
                 [0.0, 1.0, 42.0, 101.0, 160.0, 601.0],
                 [0, 1]]
    catelist = ['blacklist', 'receipt_phone_address_agreement', 'gender', 'receipt_address', 'household_register']
    typelist = []
    for i in range(len(featurelist)):
        f = featurelist[i]
        if f in catelist:
            typelist.append('cate')
        else:
            typelist.append('no_cate')

    f_g = {'feature': featurelist, 'group': grouplist, 'type': typelist}
    f_g_str = json.dumps(f_g)
    with open('feature_group_003.json', 'w') as f:
        f.write(f_g_str)


def make_feature_group_(x):

    featurelist = ['score',
                     'account_rank',
                     'deal_order_number',
                     'avg_order_amount',
                     'max_pay_amount',
                     'last_consume_days',
                     'avg_discount',
                     'earliest_consume_days',
                     'hist_consume_days',
                     'order_refund_times',
                     'phone_number',
                     'application_platform_number',
                     'application_number',
                     'apply_max_interval',
                     'blacklist',
                     'receipt_phone_address_agreement',
                     'gender',
                     'receipt_address',
                   'household_register'
    ]

    grouplist = []
    typelist = []
    catelist = ['blacklist', 'receipt_phone_address_agreement', 'gender', 'receipt_address', 'household_register']
    k = [4, 3, 3, 5, 5, 5, 4, 5, 4, 3, 3, 3, 3, 5, 2, 2, 2, 34, 34]
    for i in range(len(featurelist)):
        f = featurelist[i]
        g = make_group(x[f], k[i])
        if f in catelist:
            typelist.append('cate')
        else:
            typelist.append('no_cate')
        grouplist.append(g)

    f_g = {'feature': featurelist, 'group': grouplist, 'type': typelist}
    f_g_str = json.dumps(f_g)
    with open('feature_group_.json', 'w') as f:
        f.write(f_g_str)



def woe_train(colname, k, c):
    XX = X_train
    yy = y_train
    x = XX[colname]
    group = make_group(x, k)
    rdf = woe(x, yy, group, c)
    woe_list = rdf['woe'][rdf['min'] != 'nan']
    min_list = rdf['min'][rdf['min'] != 'nan']
    print(sum(rdf['iv']))
    plt.bar(min_list, woe_list)


def woe_test(colname, k, c):
    XX = X_test
    yy = y_test
    x = XX[colname]
    group = make_group(x, k)
    rdf = woe(x, yy, group, c)
    woe_list = rdf['woe'][rdf['min'] != 'nan']
    print(sum(rdf['iv']))
    plt.bar([i for i in range(len(woe_list))], woe_list)

if __name__ == '__main__':

    dataset = get_data()
    data = dataset.data
    label = dataset.label

    col = ['years', 'score', 'account_rank', 'deal_order_number',
            'avg_order_amount', 'max_pay_amount', 'last_consume_days', 'avg_discount', 'earliest_consume_days',
            'hist_consume_days', 'order_refund_times', 'phone_number', 'application_platform_number',
            'application_number', 'apply_max_interval', 'phone_number_rank', 'blacklist',
            'receipt_phone_address_agreement', 'nan_number', 'gender', 'receipt_address', 'household_register']
    data['label'] = label
    # data = data[data['nan_number']]

    X = data[col]
    y = pd.DataFrame(data['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    woe_list = woe_fit(X_train, y_train)

    X_train_new = woe_transform_train(X_train, woe_list)
    X_test_new = woe_transform_test(X_test, woe_list)

    dataset = bunch.Bunch(X_train=X_train_new,
                          y_train=list(y_train['label']),
                          X_test=X_test_new,
                          y_test=list(y_test['label']))
    writeBunchobj('dataset_woe.data', dataset)

    # x = X_train.account_rank
    # group = make_group(x, 5)
    # rdf = woe(x, y_train, group, 'n')
    # woe_list = rdf['woe'][rdf['min'] != 'nan']
    # print(sum(rdf['iv']))
    # plt.bar([i for i in range(len(woe_list))], woe_list)

