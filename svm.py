

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, ShuffleSplit, train_test_split, GridSearchCV
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


X = pd.read_excel('X.xlsx')
y = pd.read_excel('y.xlsx')

X_r, y_r = RandomOverSampler().fit_sample(X, y.values.ravel())
X_train, X_test, y_train, y_test = train_test_split(X_r, y_r, test_size=0.3, random_state=10)


thresholds = np.linspace(0, 0.001, 100)
# Set the parameters by cross-validation
param_grid = {'gamma': thresholds}

clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
clf.fit(X_train, y_train)
print("best param: {0}\nbest score: {1}".format(clf.best_params_,
                                                clf.best_score_))
y_pred = clf.predict(X_test)

print("查准率：", metrics.precision_score(y_pred, y_test))
print("召回率：", metrics.recall_score(y_pred, y_test))
print("F1：", metrics.f1_score(y_pred, y_test))