#ROC, classification report

import json
from sklearn.metrics import roc_curve, auc, classification_report

def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))
