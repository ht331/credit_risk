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

    def __init__(self, cost_matrix=None, w=0):
        """

        :param cost_matrix: 代价矩阵 default=[[1,1],[1,1]]
        :param w: 测试代价与误分类代价的调整参数 default=0
        """
        self._positive_label = 1  # 正类标签
        self._negative_label = 0  # 负类标签
        if cost_matrix is None:
            self.cost_matrix = np.array([[1, 2], [2, 1]])
        else:
            self.cost_matrix = cost_matrix
        if (w >= 0) and (w <= 1):
            self.w = w
        else:
            raise (' "w" should be between 0 and 1, but get value {0}'.format(w))
        global transform_tree, tree_
        transform_tree = list()
        tree_ = dict()
        self._tree = tree_
        # self._tree = self.fit(x, y)
        self.__transform_tree = transform_tree

    def fit(self, x, y):

        x = pd.DataFrame(data=x)
        y = np.array(y)
        x.reset_index(drop=True, inplace=True)
        feature_set = list(x.columns)
        self._tree = self._generate_tree(x, y, feature_set)

    def predict(self, x):

        self._transform_tree(self._tree, x)
        y_pred = np.array([0] * len(x))
        for i in range(len(self.__transform_tree)):
            y_pred[self.__transform_tree[i][0]] = self.__transform_tree[i][1]
        return y_pred

    def _cost_gini_info(self, y):
        """
            计算引入代价的Gini
        :param y:
        :return:
        """
        n_t = y.size
        if n_t != 0:

            n_1 = y[y == self._positive_label].size
            n_0 = y[y == self._negative_label].size

            type_y = self._major_class(y)
            if type_y == 0:
                prop_00 = n_0 / n_t
                prop_10 = n_1 / n_t
                # prop_list = [pow(prop_n, 2), prop_p * prop_n, prop_p * prop_n, pow(prop_p, 2)]
                # return 1 - self.cost_matrix.ravel().dot(prop_list)
                return 1 - self.cost_matrix[0][0] * pow(prop_00, 2) - self.cost_matrix[1][0] * pow(prop_10, 2)
            else:
                prop_01 = n_0 / n_t
                prop_11 = n_1 / n_t
                return 1 - self.cost_matrix[0][1] * pow(prop_01, 2) - self.cost_matrix[1][1] * pow(prop_11, 2)

        else:
            return 0
        # if n_t != 0:
        #     prop_p = n_p / n_t
        #     prop_n = n_n / n_t
        #     return 1 - pow(prop_p, 2) - pow(prop_n, 2)
        # else:
        #     return 0

    def _cost_gini_index(self, y_0, y_1, y):
        """

        :param y_0: 预测为0
        :param y_1: 预测为1
        :param y: 全部的label
        :return:
        """

        gini_t = self._cost_gini_info(y)
        gini_left = self._cost_gini_info(y_0)
        gini_right = self._cost_gini_info(y_1)
        q = y_0.size / y.size
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

        feature_list = list(set(np.array(feature_x).reshape(1, -1)[0]))
        feature_list.sort()
        split_dict = {}  # 存放划分点和对应的gicf

        if len(feature_list) == 1:
            split_point = feature_list[0]
            y_l = y
            y_r = np.array([])
            gicf_ = self._cost_gini_index(y_l, y_r, y)
            return split_point, gicf_
        else:
            for i in range(len(feature_list) - 1):
                split_point = (feature_list[i] + feature_list[i + 1]) / 2
                y_l = y[feature_x[feature_x <= split_point].index]
                y_r = y[feature_x[feature_x > split_point].index]
                gicf_ = self._cost_gini_index(y_l, y_r, y)
                split_dict.update({split_point: gicf_})
        # print(feature_list)
        return max(split_dict, key=split_dict.get), max(split_dict.values())

    def _major_class(self, y):
        """
            反互y中多数类
        :param y:
        :return:
        """
        positive_num = y[y == self._positive_label].size
        if positive_num > y.size / 2:
            node = 1
        elif positive_num < y.size / 2:
            node = 0
        else:
            if random() > 0.5:
                node = 1
            else:
                node = 0
        return node

    def _generate_tree(self, x, y, feature_set):
        """
            GART树生成器
        :param x: x 存储为pd.Dataframe
        :param y:
        :return:
        """
        # print(feature_set)
        x = pd.DataFrame(data=x)
        y = np.array(y)
        x.reset_index(drop=True, inplace=True)

        if len(set(y.reshape(1, -1)[0])) == 1:
            # 同一个类别
            return {y[0]}

        elif (len(feature_set) == 0) or (x[feature_set].drop_duplicates().shape[0] == 1):
            return {self._major_class(y)}

        else:
            split_feature = []
            for f in feature_set:
                feature_x = x[f]
                split_point, gicf = self._calc_feature_gicf(feature_x, y)
                split_feature.append([f, float(split_point), float(gicf)])
            split_feature = np.array(split_feature)
            max_gicf = max(split_feature[:, 2])
            best_feature = split_feature[:, 0][split_feature[:, 2] == max_gicf][0]
            split_point = split_feature[:, 1][split_feature[:, 2] == max_gicf][0]
            split_point = float(split_point)

            x_l = x[x[best_feature] <= split_point]
            x_r = x[x[best_feature] > split_point]

            y_l = y[x_l.index]
            y_r = y[x_r.index]
            if len(x_r) == 0:
                return {self._major_class(y_l)}
            elif len(x_l) == 0:
                return {self._major_class(y_r)}

            new_features = feature_set.copy()
            new_features.remove(best_feature)
            tree = {best_feature: {'<' + str(split_point): {}, '>' + str(split_point): {}}}
            tree[best_feature]['<' + str(split_point)] = self._generate_tree(x_l, y_l, new_features)
            tree[best_feature]['>' + str(split_point)] = self._generate_tree(x_r, y_r, new_features)
        return tree

    def _transform_tree(self, tree, x):
        """
            翻译决策树
        :param tree:
        :return:
        """
        if type(tree) is set:
            label = list(tree)[0]
            self.__transform_tree.append([x.index, label])
            return x.index, label
        else:
            feature, branch = list(tree.items())[0]
            judge_list = list(branch.keys())
            label_list = []
            label = 0
            for condition in judge_list:
                if '<' in condition:
                    condition_float = condition.replace('<', '')
                    condition_float = float(condition_float)
                    x_temp = x[x[feature] <= condition_float]
                else:
                    condition_float = condition.replace('>', '')
                    condition_float = float(condition_float)
                    x_temp = x[x[feature] > condition_float]
                label_list, label = self._transform_tree(branch.get(condition), x_temp)
            return label_list, label


if __name__ == "__main__":
    import pandas as pd
    from sklearn import metrics

    from prepare import readbunchobj

    data = readbunchobj('dataset_woe.data')
    X_train = pd.DataFrame(data.X_train)
    X_test = data.X_test
    y_train = data.y_train
    y_test = data.y_test

    print('data loaded')
    csc = CostSensitiveCart(cost_matrix=[[1, 14], [2, 1]])
    csc.fit(X_train, y_train)
    y_pred = csc.predict(X_test)

    c_m = metrics.confusion_matrix(y_test, y_pred)
    print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
    print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
    print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
    print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
    print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
