import pandas as pd
import numpy as np
from imblearn.under_sampling import NeighbourhoodCleaningRule, TomekLinks, RandomUnderSampler
from sklearn import metrics
from catboost import CatBoostClassifier, CatBoost, Pool
from prepare import readbunchobj


def model_(X, y, rs):
    osp = RandomUnderSampler(random_state=rs)
    x_train_, y_train_ = osp.fit_sample(X, y)

    xx = pd.DataFrame(X)
    xx['label'] = list(y)
    x1 = xx[xx['label'] == 1]
    x0 = xx[xx['label'] == 0]

    x00 = x0.sample()
    # 基础模型
    clf = CatBoostClassifier(loss_function='Logloss',
                             logging_level='Silent',
                             cat_features=categorical_features_indices)
    clf.fit(x_train_, y_train_)
    return clf


def bagging_boost_fit(X, y, n):
    all_clf = []
    all_score = []
    for i in range(n):
        clf = model_(X, y, i)
        all_clf.append(clf)
        s = clf.score(X, y)
        all_score.append(s)
    return all_clf, all_score


def bagging_boost_predict(X, clf_list, clf_score):
    y_pred = []
    clf_num = len(clf_list)
    for i in range(clf_num):
        clf = clf_list[i]
        clf_socre = clf_score[i]
        y_pred_ = clf.predict(X)
        y_pred.append(list(y_pred_ * clf_socre))
    y_pred_df = pd.DataFrame(data=y_pred)
    sum_y = pd.DataFrame(y_pred_df.sum(axis=0) / clf_num)
    sum_y.loc[sum_y[0] < 0.5, 0] = 0
    sum_y.loc[sum_y[0] > 0, 0] = 1
    return list(sum_y[0])


def test_model(y_test, y_pred):
    c_m = metrics.confusion_matrix(y_test, y_pred)
    print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
    print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
    print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
    print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
    print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
    print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))


if __name__ == '__main__':
    data = readbunchobj('dataset.data')
    X_train = pd.DataFrame(data.X_train)
    X_test = data.X_test
    y_train = data.y_train
    y_test = data.y_test

    categorical_features_indices = np.where((X_train.dtypes != np.float) & (X_train.dtypes != np.int64))[0]  # 类型特征的索引

    n = 50  # 做100个子样本

    clf_list, clf_score = bagging_boost_fit(X_train, y_train, n)
    y_pred = bagging_boost_predict(X_test, clf_list, clf_score)
    test_model(y_test, y_pred)
