import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import NuSVR
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE, ADASYN, RandomOverSampler
from prepare import readbunchobj
from decimal import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA

getcontext().prec = 10


def AsyBL(x_train, y_train, x_test, y_test, n_est, c1, c2, weak_clf=DecisionTreeRegressor(max_depth=1)):

    w = [1/len(y_train)] * len(y_train)
    Fx_pred = np.zeros(len(x_test))
    Fx_train = np.zeros(len(y_train))
    p = [c2 / (c1 + c2)] * len(y_train)
    scorelist = []
    nel = []
    for k in range(n_est):
        # z = [(Decimal(y_train[i]) - Decimal(p[i])) / (Decimal(p[i]) * Decimal((1 - p[i])))
        # for i in range(len(y_train))]
        z = []
        for i in range(len(y_train)):
            if p[i] == 0:
                if y_train[i] == 0:
                    z.append(-1)
                else:
                    z.append(-99)
            elif p[i] == 1:
                if y_train[i] == 0:
                    z.append(99)
                else:
                    z.append(1)
            else:
                z.append((y_train[i] - p[i]) / (p[i] * (1 - p[i])))

        weak_clf.fit(x_train, z, sample_weight=w)
        y_train_i = weak_clf.predict(x_train)
        y_pred_i = weak_clf.predict(x_test)
        Fx_pred = [(Fx_pred[i] + y_pred_i[i]) for i in range(len(x_test))]  # update ypred
        Fx_train = [(Fx_train[i] + 0.5 * y_train_i[i]) for i in range(len(y_train))]  # update F(X)

        # p = [(c2 * np.exp(Fx)) / (c1 * np.exp(-Fx) + c2 * np.exp(Fx)) for Fx in Fx_train]
        p = []
        for Fx in Fx_train:
            if Fx >= 99:
                # print(Fx)
                p.append(1)
            elif Fx <= -99:
                # print(Fx)
                p.append(1 / c1)
            else:
                p.append((c2 * np.exp(Fx)) / (c1 * np.exp(-Fx) + c2 * np.exp(Fx)))

        w = [(px * (1 - px)) for px in p]

        pred_train = (np.array(Fx_train) > 0) * 1
        s = metrics.roc_auc_score(y_train, pred_train)
        recall = metrics.recall_score(y_train, pred_train)
        precision = metrics.precision_score(y_train, pred_train)
        scorelist.append(s)
        nel.append(k)

        pred_test = (np.array(Fx_pred) > 0) * 1
        s_t = metrics.roc_auc_score(y_test, pred_test)
        recall_t = metrics.recall_score(y_test, pred_test)
        precision_t = metrics.precision_score(y_test, pred_test)
        print('iteration %s; recall %.4f; precision %.4f;'
             ' roc_auc: %.4f; test_recall %.4f; test_precision %.4f; test_roc %.4f' % (
           k, recall, precision, s, recall_t, precision_t, s_t))
    return (np.array(Fx_pred) > 0) * 1


class AsyBoostLog:

    def __init__(self, n_estermators, c1, c2, weak_clf=DecisionTreeRegressor()):

        self.n_est = n_estermators
        self.c1 = c1
        self.c2 = c2
        self.weak_clf = weak_clf

    def fit(self, X, y):
        pass


if __name__ == '__main__':
    data = readbunchobj('dataset_delstr.data')

    X_train = pd.DataFrame(data.X_train)
    X_test = data.X_test
    y_train = data.y_train
    y_test = data.y_test
    # prep = StandardScaler()
    # X_train = prep.fit_transform(X_train)
    # X_test = prep.transform(X_test)
    #
    # osp = RandomUnderSampler(random_state=10)
    # X_train, y_train = osp.fit_sample(X_train, y_train)
    n_est = 100

    c1 = (len(y_train) - sum(y_train))/sum(y_train)

    c2 = 1
    y_pred = AsyBL(X_train, y_train, X_test, y_test, n_est, c1, c2)
    c_m = metrics.confusion_matrix(y_test, y_pred)
    print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
    print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
    print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
    print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
    print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
    print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))

    # plt.plot(estlist, sc)
    # plt.show()

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_curve')
    plt.legend(loc="lower right")
    plt.show()

