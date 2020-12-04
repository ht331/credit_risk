# 随机森林方法

from sklearn.model_selection import cross_val_score, ShuffleSplit, train_test_split, GridSearchCV
# RandomForest的分类类
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from prepare import readbunchobj

import pandas as pd
from sklearn.metrics import fbeta_score, make_scorer
from sklearn import metrics

ftwo_scorer = make_scorer(fbeta_score, beta=2)


X = pd.read_excel('X.xlsx')
y = pd.read_excel('y.xlsx')

# X_smote, y_smote = SMOTE().fit_resample(X, y)
X_r, y_r = RandomOverSampler().fit_sample(X, y.values.ravel())
X_s, y_s = SMOTE().fit_sample(X, y.values.ravel())


# # 调参
# print('Start adjusting parameters')
# param_test1 = {'n_estimators': range(10, 200, 10)}
#
# param_test2 = {'max_depth': range(3, 30, 5),
#                'min_samples_split': range(50, 201, 20)}
#
# param_test3 = {'min_samples_split': range(10, 81, 10),
#                'min_samples_leaf': range(10, 60, 10)}
#
# param_test4 = {'max_features': range(3, 83, 5)}
#
#
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
# gsearch1.fit(X_r, y_r)
# # gsearch1.best_params_ = 60
#
# print('best_params:{0}  best_score:{1}'.format(gsearch1.best_params_, gsearch1.best_score_))
# best_est = gsearch1.best_params_['n_estimators']
#
#
# gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=best_est,
#                                                          min_samples_leaf=20,
#                                                          max_features='sqrt',
#                                                          oob_score=True,
#                                                          random_state=10,
#                                                          n_jobs=-1),
#                         param_grid=param_test2,
#                         scoring='recall',
#                         iid=False,
#                         return_train_score=True,
#                         cv=5)
# gsearch2.fit(X_r, y_r)
# print('best_params:{0}  best_score:{1}'.format(gsearch2.best_params_, gsearch2.best_score_))
# best_depth = gsearch2.best_params_['max_depth']
# # gsearch2.best_params_
# # {'max_depth': 13, 'min_samples_split': 50}
# # 确定最优的max_depth=9
# # min_samples_split 与决策树其他的参数存在关联，不能一起定下来
#
# gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=best_est,
#                                                          max_depth=best_depth,
#                                                          max_features='sqrt',
#                                                          oob_score=True,
#                                                          random_state=10,
#                                                          n_jobs=-1),
#                         param_grid=param_test3,
#                         scoring='recall',
#                         iid=False,
#                         cv=5)
# gsearch3.fit(X_r, y_r)
# print('best_params:{0}  best_score:{1}'.format(gsearch3.best_params_, gsearch3.best_score_))
# best_samples_leaf = gsearch3.best_params_['min_samples_leaf']
# best_samples_split = gsearch3.best_params_['min_samples_split']
# # {'min_samples_leaf': 10, 'min_samples_split': 50}
#
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
# gsearch4.fit(X_r, y_r)
# print('best_params:{0}  best_score:{1}'.format(gsearch4.best_params_, gsearch4.best_score_))
# best_features = gsearch4.best_params_['max_features']
# # {'max_features': 28}
#
#
# X_train, X_test, y_train, y_test = train_test_split(X_r, y_r, test_size=0.3, random_state=10)
# rf = RandomForestClassifier(n_estimators=best_est,
#                             max_depth=best_depth,
#                             min_samples_split=best_samples_split,
#                             min_samples_leaf=best_samples_leaf,
#                             oob_score=True,
#                             random_state=10,
#                             max_features=best_features,
#                             n_jobs=-1)
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# print("查准率：", metrics.precision_score(y_test, y_pred))
# print("召回率：", metrics.recall_score(y_test, y_pred))
# print("F1：", metrics.f1_score(y_test, y_pred))
# metrics.confusion_matrix(y_test, y_pred)
