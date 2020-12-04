import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from prepare import readbunchobj


def asy_boost(Y_train, X_train, Y_test, X_test, C1, C2, M=20, weak_clf=DecisionTreeClassifier(max_depth=1)):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    # w = [i/np.sum(Y_train) for i in Y_train]
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for j in range(M):
        # Fit a classifier with the specific weights
        weak_clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = weak_clf.predict(X_train)
        pred_test_i = weak_clf.predict(X_test)

        #
        ilist = []
        for i in range(len(Y_train)):
            if (Y_train[i] == 1) and (pred_train_i[i] != Y_train[i]):
                ilist.append(1)
            else:
                ilist.append(0)
        eps_plus = np.dot(w, ilist)

        ilist = []
        for i in range(len(Y_train)):
            if (Y_train[i] != 1) and (pred_train_i[i] != Y_train[i]):
                ilist.append(1)
            else:
                ilist.append(0)
        gamma_minus = np.dot(w, ilist)

        ilist = []
        for i in range(len(Y_train)):
            if (Y_train[i] == 1) and (pred_train_i[i] == Y_train[i]):
                ilist.append(1)
            else:
                ilist.append(0)
        gamma_plus = np.dot(w, ilist)

        ilist = []
        for i in range(len(Y_train)):
            if (Y_train[i] != 1) and (pred_train_i[i] == Y_train[i]):
                ilist.append(1)
            else:
                ilist.append(0)
        eps_minus = np.dot(w, ilist)

        # # Indicator function
        # print("weak_clf_%02d train acc: %.4f"
        #  % (i + 1, 1 - sum(miss) / n_train))
        #
        # # Error
        # err_m = np.dot(w, miss)

        # Alpha
        # alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        alpha_m = 0.5 * np.log((gamma_plus * C1 + eps_minus * C2) / (eps_plus * C1 + gamma_minus * C2))
        # print(alpha_m)

        yy = [int(x) for x in (pred_train_i != Y_train)]
        yyy = [x if x == 1 else -1 for x in yy]
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in yyy]))
        w = w / sum(w)

        pred_test = pred_test + np.multiply(alpha_m, pred_test_i)

    pred_test = (pred_test > 0) * 1
    return pred_test


if __name__ == '__main__':
    data = readbunchobj('dataset_woe.data')
    X_train = pd.DataFrame(data.X_train)
    X_test = data.X_test
    y_train = data.y_train
    y_test = data.y_test

    # osp = RandomUnderSampler(random_state=10)
    # osp = SMOTE(random_state=10)
    # X_train, y_train = osp.fit_sample(X_train, y_train)  # SMOTE
    #
    # clf = AdaBoostClassifier(n_estimators=20)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    C1 = 10
    C2 = 1
    y_pred = asy_boost(y_train, X_train, y_test, X_test, C1, C2, M=100)

    c_m = metrics.confusion_matrix(y_test, y_pred)
    print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
    print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
    print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
    print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
    print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
    print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))
