
# GBDT

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.decomposition import PCA

from prepare import readbunchobj
data = readbunchobj('dataset_delstr.data')
X_train = pd.DataFrame(data.X_train)
X_test = data.X_test
y_train = data.y_train
y_test = data.y_test


# 非平衡样本处理
# osp = SMOTE(random_state=10)
# osp = BorderlineSMOTE(random_state=10)
osp = RandomUnderSampler(random_state=10)
X_train, y_train = osp.fit_sample(X_train, y_train)  # SMOTE


# loss='auto', learning_rate=0.1, max_iter=100, max_leaf_nodes=31,
# max_depth=None, min_samples_leaf=20, l2_regularization=0.0,
# max_bins=255, warm_start=False, scoring=None,
# validation_fraction=0.1, n_iter_no_change=None,
# tol=1e-07, verbose=0, random_state=None

param1 = {'learning_rate': np.arange(0.01, 0.2, 0.01)}
param2 = {'max_iter': range(10, 30, 5)}
param3 = {'max_leaf_nodes': range(2, 20, 1)}
param4 = {'max_depth': range(1, 30, 1)}
param5 = {'min_samples_leaf': range(10, 101, 10)}
param6 = {'l2_regularization': np.arange(0.0, 2, 0.1)}

scoring = 'roc_auc'

gsearch1 = GridSearchCV(estimator=
                        HistGradientBoostingClassifier(
                            random_state=10,
                        ),
                        param_grid=param1,
                        scoring=scoring,
                        n_jobs=-1,
                        cv=5)
gsearch1.fit(X_train, y_train)
print('best_params:{0}  best_score:{1}'.format(gsearch1.best_params_, gsearch1.best_score_))
lr = gsearch1.best_params_['learning_rate']
# best_params:{'learning_rate': 0.6}  best_score:0.9512231982157461

gsearch2 = GridSearchCV(estimator=
                        HistGradientBoostingClassifier(
                            random_state=10,
                            learning_rate=lr,
                        ),
                        param_grid=param2,
                        scoring=scoring,
                        n_jobs=-1,
                        cv=5)
gsearch2.fit(X_train, y_train)
print('best_params:{0}  best_score:{1}'.format(gsearch2.best_params_, gsearch2.best_score_))
mi = gsearch2.best_params_['max_iter']
# best_params:{'max_iter': 200}  best_score:0.9644651997338606

gsearch3 = GridSearchCV(estimator=
                        HistGradientBoostingClassifier(
                            random_state=10,
                            learning_rate=lr,
                            max_iter=mi,
                        ),
                        param_grid=param3,
                        scoring=scoring,
                        n_jobs=-1,
                        cv=5)
gsearch3.fit(X_train, y_train)
print('best_params:{0}  best_score:{1}'.format(gsearch3.best_params_, gsearch3.best_score_))
mln = gsearch3.best_params_['max_leaf_nodes']
# best_params:{'max_leaf_nodes': 100}  best_score:0.9770635523808995

gsearch4 = GridSearchCV(estimator=
                        HistGradientBoostingClassifier(
                            random_state=10,
                            learning_rate=lr,
                            max_iter=mi,
                            max_leaf_nodes=mln
                        ),
                        param_grid=param4,
                        scoring=scoring,
                        n_jobs=-1,
                        cv=5)
gsearch4.fit(X_train, y_train)
print('best_params:{0}  best_score:{1}'.format(gsearch4.best_params_, gsearch4.best_score_))
md = gsearch4.best_params_['max_depth']
# best_params:{'max_depth': 20}  best_score:0.9775816715080079

gsearch5 = GridSearchCV(estimator=
                        HistGradientBoostingClassifier(
                            random_state=10,
                            learning_rate=lr,
                            max_iter=mi,
                            max_leaf_nodes=mln,
                            max_depth=md,
                        ),
                        param_grid=param5,
                        scoring=scoring,
                        n_jobs=-1,
                        cv=5)
gsearch5.fit(X_train, y_train)
print('best_params:{0}  best_score:{1}'.format(gsearch5.best_params_, gsearch5.best_score_))
msl = gsearch5.best_params_['min_samples_leaf']
# best_params:{'min_samples_leaf': 20}  best_score:0.9775816715080079


gsearch6 = GridSearchCV(estimator=
                        HistGradientBoostingClassifier(
                            random_state=10,
                            learning_rate=lr,
                            max_iter=mi,
                            max_leaf_nodes=mln,
                            max_depth=md,
                            min_samples_leaf=msl,
                        ),
                        param_grid=param6,
                        scoring=scoring,
                        n_jobs=-1,
                        cv=5)
gsearch6.fit(X_train, y_train)
print('best_params:{0}  best_score:{1}'.format(gsearch6.best_params_, gsearch6.best_score_))
l2r = gsearch6.best_params_['l2_regularization']
# best_params:{'l2_regularization': 0.30000000000000004}  best_score:0.9780450886460196


hgdbt = HistGradientBoostingClassifier(
                            random_state=10,
                            learning_rate=lr,
                            max_iter=mi,
                            max_leaf_nodes=mln,
                            max_depth=md,
                            min_samples_leaf=msl,
                            l2_regularization=l2r
                        )
hgdbt.fit(X_train, y_train)
y_pred = hgdbt.predict(X_test)
c_m = metrics.confusion_matrix(y_test, y_pred)
print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))
