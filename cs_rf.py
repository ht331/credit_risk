# coding:utf-8
"""
cost sensitive random forest


"""
import pandas as pd
import numpy as np
from random import random


class CostSensitiveCart:
    """
        引入代价敏感分析的CART决策树
    """
    TREE = {}

    def __init__(self, cost_matrix=None, w=0):
        """

        :param cost_matrix: 代价矩阵 default=[[1,1],[1,1]]
        :param w: 测试代价与误分类代价的调整参数 default=0
        """
        self.positive_label = 1  # 正类标签
        self.negative_label = 0  # 负类标签
        if cost_matrix is None:
            self.cost_matrix = np.array([[1, 1], [1, 1]])
        else:
            self.cost_matrix = cost_matrix
        if (w >= 0) and (w <= 1):
            self.w = w
        else:
            raise(' "w" should be between 0 and 1, but get value {0}'.format(w))

    def fit(self, x, y):

        feature_set = x.columns
        self._generate_tree(x, y, feature_set)

    def predict(self, x):
        return None

    def _cost_gini_info(self, y):
        """
            计算引入代价的Gini
        :param y:
        :return:
        """
        n_t = y.size
        n_p = y[y == self.positive_label].size
        n_n = y[y == self.negative_label].size
        prop_p = n_p / n_t
        prop_n = n_n / n_t
        prop_list = [pow(prop_n, 2), prop_p * prop_n, pow(prop_p, 2), prop_p * prop_n]
        return self.cost_matrix.ravel().dot(prop_list)

    def _cost_gini_index(self, y_l, y_r, y):
        """

        :param y_l: 左孩子的label
        :param y_r: 右孩子的label
        :param y: 全部的label
        :return:
        """

        gini_t = self._cost_gini_info(y)
        gini_left = self._cost_gini_info(y_l)
        gini_right = self._cost_gini_info(y_r)
        q = y_l.size / y.size
        return gini_t - q * gini_left - (1 - q) * gini_right

    def _calc_gicf(self, y_l, y_r, y, ca):
        """
            计算gini代价函数
        :param y_l: 同上
        :param y_r: 同上
        :param y: 同上
        :param ca: 某属性下的测试误差
        :return:
        """
        delta_gini = self._cost_gini_index(y_l, y_r, y)
        return (pow(2, delta_gini) - 1) / (pow((ca + 1), self.w))

    def _calc_feature_gicf(self, feature_x, y):
        """
            计算feature特征下的所有可能划分的gini代价函数
        :param feature_x: x数据集在feature下的取值
        :param y:对于feature_x的label
        :param ca: feature属性对应的测试代价
        :return:
        """

        feature_x = pd.DataFrame(data=feature_x)
        feature_list = list(set(list(feature_x)))
        feature_list.sort()
        split_dict = {}  # 存放划分点和对应的gicf
        for i in range(len(feature_list) - 1):
            split_point = (feature_list[i] + feature_list[i + 1]) / 2
            y_l = y[feature_x[feature_x[0] <= split_point].index]
            y_r = y[feature_x[feature_x[0] > split_point].index]
            gicf_ = self._cost_gini_index(y_l, y_r, y)
            split_dict.update({split_point: gicf_})
        return max(split_dict, key=split_dict.get), max(split_dict.values())

    def _generate_tree(self, x, y, feature_set):
        """
            GART树生成器
        :param x: x 存储为pd.Dataframe
        :param y:
        :return:
        """
        x = pd.DataFrame(data=x)
        node = []  # node = ['feature', split_point, 0, 1]  以此为结点对应的属性、划分点、左孩子对应的label、右孩子对应的label
        if len(set(y)) == 1:
            # 同一个类别
            node = [feature_set, None, list(set(y))[0], list(set(y))[0]]
            return node

        elif (len(feature_set) == 0) or (x.drop_duplicates()[0].size == 1):
            positive_num = y[y == self.positive_label].size
            if positive_num > y.size/2:
                node = [feature_set, None, 1, 1]
            elif positive_num < y.size/2:
                node = [feature_set, None, 0, 0]
            else:
                if random() > 0.5:
                    node = [feature_set, None, 1, 1]
                else:
                    node = [feature_set, None, 0, 0]
            return node

        else:
            split_feature = []
            for f in feature_set:
                feature_x = x[f]
                split_point, gicf = self._calc_feature_gicf(feature_x, y)
                split_feature.append([f, split_point, gicf])
            split_feature = np.array(split_feature)
            max_gicf = max(split_feature[:, 2])
            best_feature = split_feature[:, 0][split_feature[:, 2] == max_gicf][0]
            split_point = split_feature[:, 1][split_feature[:, 2] == max_gicf][0]

            x_l = x[x[best_feature] <= split_point]
            x_r = x[x[best_feature] > split_point]
            y_l = y[x_l.index]
            y_r = y[x_r.index]
            xy_list = [[x_l, y_l], [x_r, y_r]]
            for xy in xy_list:
                xx = xy[0]
                yy = xy[1]
                if len(xx) == 0:
                    return node
                else:
                    feature_set_ = feature_set.copy()
                    feature_set_.remove(best_feature)
                    node = self._generate_tree(xx, yy, feature_set_)

