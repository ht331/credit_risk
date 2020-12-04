# 代价敏感随机森林

from random import random, sample, randint
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


class CostSensitiveRandomForest:

    def __init__(self,
                 n_estimators=100,
                 random_state=10,
                 cost=10,
                 max_depth=10):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.cost = cost
        # np.random.set_state(self.random_state)
        global est
        est = pd.DataFrame()
        self._est = est

    def fit(self, x, y):
        self._est = self._model_fit(x, y)

    def predict(self, x):
        y_temp = []
        for i in range(len(self._est)):
            e = self._est[0][i]
            col = self._est[2][i]
            y_temp_ = e.predict(x[col])
            y_temp.append(y_temp_)
        y_df = pd.DataFrame(y_temp)
        y_np = np.array(y_df.sum())
        y_np[y_np < self.n_estimators] = 0
        y_np[y_np >= self.n_estimators] = 1
        return y_np

    def _bootstrap_sampling(self, x, y):
        # 有放回抽样 产生一个x.size的样本
        random_index = np.random.randint(0, len(x), size=(self.n_estimators * 2, len(x)))
        sample_set = []
        for _ in random_index:
            test_index = [i for i in range(len(x)) if i not in _]
            x_train_sample = x.iloc[_]
            x_test_sample = x.iloc[test_index]
            x_train_sample.reset_index(drop=True, inplace=True)
            x_test_sample.reset_index(drop=True, inplace=True)
            y_train_sample = y[pd.Index(_)]
            y_test_sample = y[pd.Index(test_index)]
            sample_set.append([x_train_sample, y_train_sample, x_test_sample, y_test_sample])
        return sample_set

    def _calc_avg_cost(self, y_true, y_pred):
        c_m = metrics.confusion_matrix(y_true, y_pred)
        return (1 * c_m[0][1] + self.cost * c_m[1][0]) / (sum(c_m.ravel()))

    def _sort_score(self, estimators_list):
        df_ = pd.DataFrame(data=estimators_list)
        df_.sort_values(by=1, inplace=True)
        df_.reset_index(drop=True, inplace=True)

        return df_[:self.n_estimators]

    def _model_fit(self, x, y):
        sample_set = self._bootstrap_sampling(x, y)
        estimators_list = []
        for i in range(2 * self.n_estimators):
            random_feature_num = randint(1, self.max_depth)
            sample_x_train = sample_set[i][0]
            sample_x_col = sample(list(sample_x_train.columns), random_feature_num)
            sample_x_train = sample_x_train[sample_x_col]
            sample_y_train = sample_set[i][1]
            x_test = sample_set[i][2][sample_x_col]
            y_test = sample_set[i][3]
            estimators = DecisionTreeClassifier()
            estimators.fit(sample_x_train, sample_y_train)
            y_pred = estimators.predict(x_test)
            cost_score = self._calc_avg_cost(y_test, y_pred)
            estimators_list.append([estimators, cost_score, sample_x_col])
            # print(cost_score)
        best_estimators = self._sort_score(estimators_list)
        return best_estimators

    def _vote(self, y_sum):
        pass


if __name__ == '__main__':
    import pandas as pd
    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    from prepare import readbunchobj

    data = readbunchobj('dataset_woe.data')
    X_train = pd.DataFrame(data.X_train)
    X_test = data.X_test
    y_train = data.y_train
    y_test = data.y_test

    print('load')
    X_train = pd.DataFrame(data=X_train)
    y_train = np.array(y_train)
    X_test = pd.DataFrame(data=X_test)
    rf = CostSensitiveRandomForest(n_estimators=100, cost=50, max_depth=19)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    c_m = metrics.confusion_matrix(y_test, y_pred)
    print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
    print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
    print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
    print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
    print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))

    print("=" * 40)
    rf0 = RandomForestClassifier()
    rf0.fit(X_train, y_train)
    y_pred = rf0.predict(X_test)
    c_m = metrics.confusion_matrix(y_test, y_pred)
    print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
    print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
    print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
    print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
    print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
