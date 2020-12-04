# nan:mean
# over_sample: class_weight
# RF
# estimator: roc_auc

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier


from prepare import readbunchobj
data = readbunchobj('dataset_woe.data')
X_train = pd.DataFrame(data.X_train)
X_test = data.X_test
y_train = data.y_train
y_test = data.y_test

# osp = RandomUnderSampler(random_state=10)

osp = BorderlineSMOTE()
X_train, y_train = osp.fit_sample(X_train, y_train)  # SMOTE

# fsel = SelectFromModel(GradientBoostingClassifier())
# X_train = fsel.fit_transform(X_train, y_train)
# X_test = fsel.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
c_m = metrics.confusion_matrix(y_test, y_pred)
print('真反例:{0}\n假反例:{1}\n真正例:{2}\n假正例:{3}\n'.format(c_m[0][0], c_m[1][0], c_m[1][1], c_m[0][1]))
print("召回率:%.4f" % metrics.recall_score(y_test, y_pred))
print("查准率:%.4f" % metrics.precision_score(y_test, y_pred))
print("F1：%.4f" % metrics.f1_score(y_test, y_pred))
print("roc_auc:%.4f" % metrics.roc_auc_score(y_test, y_pred))
print("F-measure:%.4f" % (metrics.recall_score(y_test, y_pred) * metrics.precision_score(y_test, y_pred)))

# from sklearn.metrics import roc_curve, auc
# fpr, tpr, threshold = roc_curve(y_test, y_pred)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC_curve')
# plt.legend(loc="lower right")
# plt.show()