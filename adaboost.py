import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule, TomekLinks
from imblearn.over_sampling import SMOTE
from prepare import readbunchobj


def my_adaboost_clf(Y_train, X_train, Y_test, X_test, M=20, weak_clf=DecisionTreeClassifier(max_depth = 1)):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(M):
        # Fit a classifier with the specific weights
        weak_clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = weak_clf.predict(X_train)
        pred_test_i = weak_clf.predict(X_test)

        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        print("weak_clf_%02d train acc: %.4f"
         % (i + 1, 1 - sum(miss) / n_train))

        # Error
        err_m = np.dot(w, miss)

        # Alpha
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))

        # New weights
        miss2 = [x if x==1 else -1 for x in miss] # -1 * y_i * G(x_i): 1 / -1
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        w = w / sum(w)

        # Add to prediction
        pred_train_i = [1 if x == 1 else -1 for x in pred_train_i]
        pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]
        pred_train = pred_train + np.multiply(alpha_m, pred_train_i)
        pred_test = pred_test + np.multiply(alpha_m, pred_test_i)

    pred_train = (pred_train > 0) * 1
    pred_test = (pred_test > 0) * 1

    print("My AdaBoost clf train accuracy: %.4f" % (sum(pred_train == Y_train) / n_train))
    print("My AdaBoost clf test accuracy: %.4f" % (sum(pred_test == Y_test) / n_test))
    return pred_test


if __name__ == '__main__':
    data = readbunchobj('dataset_woe.data')
    X_train = pd.DataFrame(data.X_train)
    X_test = data.X_test
    y_train = data.y_train
    y_test = data.y_test

    # osp = SMOTE(random_state=10)
    osp = RandomUnderSampler(random_state=10)
    X_train, y_train = osp.fit_sample(X_train, y_train)  # SMOTE

    clf = AdaBoostClassifier(n_estimators=100, random_state=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # y_pred = my_adaboost_clf(y_train, X_train, y_test, X_test, M=100)

    c_m = metrics.confusion_matrix(y_test, y_pred)
    print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
    print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
    print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
    print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
    print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
    print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))
