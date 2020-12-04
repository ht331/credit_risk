# one class svm


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


# 读取数据
data = pd.read_excel('data_set.xlsx')
col = data.columns
X = data[col[:19]]
y = data[col[-1]]
# 训练集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
# 缺失值插补
imp = SimpleImputer(strategy='mean')  # 均值 单变量插补
X_train = imp.fit_transform(X_train)  # 训练集插补
X_test = imp.transform(X_test)  # 测试集插补

# 归一化
prep = StandardScaler()
X_train = prep.fit_transform(X_train)
X_test = prep.transform(X_test)

ocs = OneClassSVM()
ocs.fit(X_train)
y_pred = ocs.predict(X_test)
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

c_m = metrics.confusion_matrix(y_test, y_pred)
print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
