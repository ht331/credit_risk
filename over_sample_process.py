
import pandas as pd
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SMOTENC, SVMSMOTE, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def estimator_(X_, y_, X_t):
    rf = RandomForestClassifier(random_state=10, n_jobs=-1)
    rf.fit(X_, y_)
    y_pred = rf.predict(X_t)
    c_m = metrics.confusion_matrix(y_test, y_pred)
    print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
    print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
    print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
    print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
    print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))


if __name__ == "__main__":
    data = pd.read_excel('data_set.xlsx')
    col = data.columns
    X = data[col[:19]]
    y = data[col[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    imp = SimpleImputer(strategy='mean')  # 均值 单变量插补
    X_train = imp.fit_transform(X_train)  # 训练集插补
    X_test = imp.transform(X_test)  # 测试集插补

    prep = StandardScaler()
    X_train = prep.fit_transform(X_train)
    X_test = prep.transform(X_test)

    ops_ada = ADASYN(random_state=10)
    ops_bsmote = BorderlineSMOTE(random_state=10)
    ops_ksmote = KMeansSMOTE(random_state=10)
    ops_rs = RandomOverSampler(random_state=10)
    ops_s = SMOTE(random_state=10)

    X_train_ada, y_train_ada = ops_ada.fit_sample(X_train, y_train)
    X_train_bsmote, y_train_bsmote = ops_bsmote.fit_sample(X_train, y_train)
    X_train_ksmote, y_train_ksmote = ops_ksmote.fit_sample(X_train, y_train)
    X_train_rs, y_train_rs = ops_rs.fit_sample(X_train, y_train)
    X_train_s, y_train_s = ops_s.fit_sample(X_train, y_train)

    dic_ = {'ADASYN': [X_train_ada, y_train_ada],
            'BorderlineSMOTE': [X_train_bsmote, y_train_bsmote],
            'RandomOverSampler': [X_train_rs, y_train_rs],
            'SMOTE': [X_train_s, y_train_s]}

    for t in dic_.keys():
        print('over sampler: %s \n' % t)
        X_ = dic_[t][0]
        y_ = dic_[t][1]
        X_t = X_test
        estimator_(X_, y_, X_t)
