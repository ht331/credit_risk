# nan:mean
# over_sample: SMOTE
# RF
# estimator: recall

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from prepare import readbunchobj

# 读取数据
# data = pd.read_excel('data_set.xlsx')
# data = data[data['num_na'] < 10]
# col = data.columns
# X = data[col[:-1]]
# y = data[col[-1]]
# 训练集划分

data = readbunchobj('dataset_woe.data')
X_train = pd.DataFrame(data.X_train)
X_test = data.X_test
y_train = data.y_train
y_test = data.y_test
# # 缺失值插补
# imp = SimpleImputer(strategy='mean')  # 均值 单变量插补
# X_train = imp.fit_transform(X_train)  # 训练集插补
# X_test = imp.transform(X_test)  # 测试集插补

# # 归一化
# prep = StandardScaler()
# X_train = prep.fit_transform(X_train)
# X_test = prep.transform(X_test)

# # pca 特征
# fea = FeatureAgglomeration(n_clusters=10)
# X_train = fea.fit_transform(X_train)
# # X_test = fea.transform(X_test)
# pca = PCA(n_components=0.9, svd_solver='full')
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)


# 非平衡样本处理
# osp = SMOTE()
# osp = BorderlineSMOTE(random_state=10)
#
# X_train, y_train = osp.fit_sample(X_train, y_train)  # SMOTE
#
# 基本模型
rf0 = RandomForestClassifier(random_state=10, n_jobs=-1)
rf0.fit(X_train, y_train)

y_pred = rf0.predict(X_test)
c_m = metrics.confusion_matrix(y_test, y_pred)
print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))


# # # Adaboost
# ad0 = AdaBoostClassifier(random_state=10)
# ad0.fit(X_train, y_train)
# y_pred = ad0.predict(X_test)
# c_m = metrics.confusion_matrix(y_test, y_pred)
# print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
# print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
# print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
# print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
# print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
# print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))

#
# # GDBT
# gdbt = GradientBoostingClassifier(random_state=10)
# gdbt.fit(X_train, y_train)
# y_pred = gdbt.predict(X_test)
# c_m = metrics.confusion_matrix(y_test, y_pred)
# print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
# print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
# print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
# print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
# print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
# print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))
#
#
# # 模型调优 超参数网格搜索
# print('Start adjusting parameters')
#
# param_test1 = {'n_estimators': range(10, 200, 10)}
# gsearch1 = GridSearchCV(estimator=
#                         RandomForestClassifier(min_samples_split=100,
#                                                min_samples_leaf=20,
#                                                max_depth=8,
#                                                max_features='sqrt',
#                                                random_state=10,
#                                                n_jobs=-1),
#                         param_grid=param_test1,
#                         scoring='recall',
#                         cv=5)
# gsearch1.fit(X_train, y_train)
# print('best_params:{0}  best_score:{1}'.format(gsearch1.best_params_, gsearch1.best_score_))
# best_est = gsearch1.best_params_['n_estimators']
# # best_params:{'n_estimators': 60}  best_score:0.8535885167464115
#
#
# param_test2 = {'max_depth': range(10, 50, 5)}
# gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=best_est,
#                                                          min_samples_leaf=20,
#                                                          max_features='sqrt',
#                                                          oob_score=True,
#                                                          random_state=10,
#                                                          n_jobs=-1),
#                         param_grid=param_test2,
#                         scoring='recall',
#                         return_train_score=True,
#                         cv=5)
# gsearch2.fit(X_train, y_train)
# print('best_params:{0}  best_score:{1}'.format(gsearch2.best_params_, gsearch2.best_score_))
# best_depth = gsearch2.best_params_['max_depth']
# # best_params:{'max_depth': 35}  best_score:0.9028885344674817
#
#
# param_test3 = {'min_samples_split': range(10, 81, 10),
#                'min_samples_leaf': range(10, 60, 10)}
# gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=best_est,
#                                                          max_depth=best_depth,
#                                                          max_features='sqrt',
#                                                          oob_score=True,
#                                                          random_state=10,
#                                                          n_jobs=-1),
#                         param_grid=param_test3,
#                         scoring='recall',
#                         cv=5)
# gsearch3.fit(X_train, y_train)
# print('best_params:{0}  best_score:{1}'.format(gsearch3.best_params_, gsearch3.best_score_))
# best_samples_leaf = gsearch3.best_params_['min_samples_leaf']
# best_samples_split = gsearch3.best_params_['min_samples_split']
# # best_params:{'min_samples_leaf': 10, 'min_samples_split': 10}  best_score:0.9143717880559986
#
#
# param_test4 = {'max_features': range(3, 83, 5)}
# gsearch4 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=best_est,
#                                                          max_depth=best_depth,
#                                                          min_samples_split=best_samples_split,
#                                                          min_samples_leaf=best_samples_leaf,
#                                                          oob_score=True,
#                                                          random_state=10,
#                                                          n_jobs=-1),
#                         param_grid=param_test4,
#                         scoring='recall',
#                         cv=5)
# gsearch4.fit(X_train, y_train)
# print('best_params:{0}  best_score:{1}'.format(gsearch4.best_params_, gsearch4.best_score_))
# best_features = gsearch4.best_params_['max_features']
# # best_params:{'max_features': 18}  best_score:0.9147970937444623
#
#
# # 调参后的模型
# rf1 = RandomForestClassifier(n_estimators=60,
#                              max_depth=35,
#                              min_samples_split=10,
#                              min_samples_leaf=10,
#                              oob_score=True,
#                              random_state=10,
#                              n_jobs=-1)
# rf1.fit(X_train, y_train)
# y_pred = rf1.predict(X_test)
# c_m = metrics.confusion_matrix(y_test, y_pred)
#
# print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
# print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
# print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
# print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
# print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
