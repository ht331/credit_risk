#导包
import numpy as np
import math
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from scipy import stats


# 求woe值和iv值
def woe(X, y, event):
    res_woe = []  # 列表存放woe字典
    res_iv = []  # 列表存放iv
    X1 = feature_discretion(X)  # 对连续型特征进行处理
    for i in range(0, X1.shape[-1]):  # 遍历所有特征
        x = X1[:, i]  # 单个特征
        woe_dict, iv1 = woe_single_x(x, y, event)  # 计算单个特征的woe值
        res_woe.append(woe_dict)
        res_iv.append(iv1)
    return np.array(res_woe), np.array(res_iv)  # 返回数组


# 求单个特征的woe值
def woe_single_x(x, y, event):
    event_total, non_event_total = count_binary(y, event)  # 计算好人坏人总数
    x_labels = np.unique(x)  # 特征中的分段
    woe_dict = {}  # 存放每个分段的名称 以及 其对应的woe值
    iv = 0
    for x1 in x_labels:  # 遍历每个分段
        y1 = y[np.where(x == x1)[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        rate_event = 1.0 * event_count / event_total
        rate_non_event = 1.0 * non_event_count / non_event_total
        # woe无穷大时处理
        if rate_event == 0:
            print()  # print("{'",x1,"'}"+":全是好人") #只输出不做处理
        elif rate_non_event == 0:
            print()  # print("{'",x1,"'}"+":全是坏人")
        else:
            woe1 = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
    return woe_dict, iv


# 计算个数
def count_binary(a, event):
    event_count = (a == event).sum()
    non_event_count = a.shape[-1] - event_count
    return event_count, non_event_count


# 判断特征数据是否为离散型
def feature_discretion(X):
    temp = []
    for i in X.columns:
        x = X[i]
        x_type = type_of_target(x)
        if x_type == 'continuous':
            x1 = discrete(x)
            temp.append(x1)
        else:
            temp.append(x)
    return np.array(temp).T


# 对连续型特征进行离散化
def discrete(x):
    res = np.array([0] * x.shape[-1], dtype=int)
    for i in range(20):
        point1 = stats.scoreatpercentile(x, i * 5)
        point2 = stats.scoreatpercentile(x, (i + 1) * 5)
        x1 = x[(x >= point1) & (x <= point2)]
        mask = np.in1d(x, x1)
        res[mask] = (i + 1)
    return res

