from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import numpy as np


data = load_breast_cancer()
x = pd.DataFrame(data['data'], columns=data['feature_names'])
y = data['target']
clf = LogisticRegression(solver='liblinear')
clf.fit(x, y)
y_pred = clf.predict_proba(x)[:, 1]

lr_fi_true = LogisticRegression(fit_intercept=True)
lr_fi_false = LogisticRegression(fit_intercept=False)
def plot_roc(y_true, y_pred):
    plt.figure(figsize=(10, 8))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label='ROC curve ')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def logloss(y_true, y_pred):
    return (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
# Рассчитать roc auc и logloss для fit_intercept = True или False
#def roc_a



#def log_loss():
y_true = [0, 0, 1, 0, 1, 1, 1]
y_pred = [1, 2, 3, 4, 5, 6, 7]
plot_roc(y_true, y_pred)
logloss(y_true, y_pred)
#auc = roc_auc_score(y_true, y_pred)
#print('AUC: %.2f' % auc)


#(tpr * fpr) / 2 + (tpr * ( 1 - fpr)) + (1-tpr)*(1-fpr) / 2 == (1 + tpr - fpr) / 2