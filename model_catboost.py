
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import NeighbourhoodCleaningRule, TomekLinks, RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from catboost import CatBoostClassifier, CatBoost, Pool
from sklearn.decomposition import PCA

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from prepare import readbunchobj
data = readbunchobj('dataset_woe.data')
X_train = pd.DataFrame(data.X_train)
X_test = data.X_test
y_train = data.y_train
y_test = data.y_test

categorical_features_indices = np.where((X_train.dtypes != np.float) & (X_train.dtypes != np.int64))[0]  # 类型特征的索引


# 非平衡样本处理
# osp = SMOTE()
# osp = BorderlineSMOTE(random_state=10)
osp = RandomUnderSampler(random_state=10)
X_train, y_train = osp.fit_sample(X_train, y_train)

# 基础模型
clf = CatBoostClassifier(loss_function='Logloss',
                         logging_level='Silent',
                         random_state=10,
                         )
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
c_m = metrics.confusion_matrix(y_test, y_pred)
print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))
#
# feature_score = clf.get_feature_importance()
# col = data.col
# plt.figure()
# plt.barh(col, feature_score, height=0.5)
# plt.show()
#

#
# # #
# param1 = {'class_weights': [[1, i] for i in range(10, 30, 1)]}  # learning_rate
# # # param2 = {'max_depth': range(5, 19, 1)}  # 树的深度
# # # param3 = {'reg_lambda': range(1, 10, 1)}  # L2正则化系数
# # # param4 = {'num_boost_round': range(100, 1000, 10)}  # 解决ml问题的树的最大数量
# # #
# # #
# scoring = 'roc_auc'
#
# gsearch1 = GridSearchCV(estimator=
#                         CatBoostClassifier(
#                             random_state=10,
#                             loss_function='Logloss',
#                             logging_level='Silent',
#                             cat_features=categorical_features_indices,
#                             thread_count=-1
#                         ),
#                         param_grid=param1,
#                         scoring=scoring,
#                         n_jobs=-1,
#                         cv=5)
# gsearch1.fit(X_train, y_train)
# print('best_params:{0}  best_score:{1}'.format(gsearch1.best_params_, gsearch1.best_score_))
# lr = gsearch1.best_params_['eta']
#
# #
# # gsearch2 = GridSearchCV(estimator=
# #                         CatBoostClassifier(
# #                             eta=lr,
# #                             random_state=10,
# #                             loss_function='Logloss',
# #                             logging_level='Silent',
# #                             cat_features=categorical_features_indices
# #                         ),
# #                         param_grid=param2,
# #                         scoring=scoring,
# #                         n_jobs=-1,
# #                         cv=5)
# # gsearch2.fit(X_train, y_train)
# # print('best_params:{0}  best_score:{1}'.format(gsearch2.best_params_, gsearch2.best_score_))
# # md = gsearch2.best_params_['max_depth']
# md = 20
#
# gsearch3 = GridSearchCV(estimator=
#                         CatBoostClassifier(
#                             eta=lr,
#                             random_state=10,
#                             loss_function='Logloss',
#                             logging_level='Silent',
#                             cat_features=categorical_features_indices,
#                             thread_count=-1
#
#                         ),
#                         param_grid=param3,
#                         scoring=scoring,
#                         n_jobs=-1,
#                         cv=5)
# gsearch3.fit(X_train, y_train)
# print('best_params:{0}  best_score:{1}'.format(gsearch3.best_params_, gsearch3.best_score_))
# l2 = gsearch3.best_params_['reg_lambda']
#
#
# gsearch4 = GridSearchCV(estimator=
#                         CatBoostClassifier(
#                             eta=lr,
#                             reg_lambda=l2,
#                             random_state=10,
#                             loss_function='Logloss',
#                             logging_level='Silent',
#                             cat_features=categorical_features_indices,
#                             thread_count=-1
#
#                         ),
#                         param_grid=param4,
#                         scoring=scoring,
#                         n_jobs=-1,
#                         cv=5)
# gsearch4.fit(X_train, y_train)
# print('best_params:{0}  best_score:{1}'.format(gsearch4.best_params_, gsearch4.best_score_))
# net = gsearch4.best_params_['num_boost_round']
#
#
# cbc = CatBoostClassifier(eta=lr,
#                          reg_lambda=l2,
#                          num_boost_round=net,
#                          random_state=10,
#                          loss_function='Logloss',
#                          logging_level='Silent',
#                          cat_features=categorical_features_indices,
#                          thread_count=-1
#
#                          )
#
#
# cbc.fit(X_train, y_train)
# y_pred = cbc.predict(X_test)
#
# c_m = metrics.confusion_matrix(y_test, y_pred)
# print('=' * 40)
# print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
# print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
# print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
# print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
# print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
# print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))
#

