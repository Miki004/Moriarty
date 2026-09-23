import numpy as np
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, precision_score, recall_score

pos_label, neg_label = 1, 0

GAIN_MATRIX = np.array([
    [0, -1],   # false positive: unnecessary maintenance check
    [-5, 0],   # false negative: missed failure (most expensive)
])


def credit_gain_score(y_true, y_pred, neg_label=neg_label, pos_label=pos_label):
    cm = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
    return np.sum(cm * GAIN_MATRIX)


gain_scorer = make_scorer(credit_gain_score)


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)


METRIC_FNS = {
    "f1": f1,
    "precision": precision,
    "recall": recall,
    "gain": credit_gain_score,
}