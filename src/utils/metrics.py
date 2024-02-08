import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np


def get_metrics(y_true, y_pred):
    
    accuracy = accuracy_score(y_true, y_pred)
    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred)
    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, y_pred)

    return [accuracy, precision, recall, f1]

def evaluation(x, y):
    """Metrics."""
    pred = np.argmax(x, axis=1)
    label = np.argmax(y, axis=1)
    metrics_orig_imege = get_metrics(pred, label)
    
    return metrics_orig_imege